# SPDX-License-Identifier: Apache-2.0
"""Conservative terminal-consumer MoE compaction outside captured graphs."""

import torch


def compact_terminal_moe(mlp, x, residual, post, comb, pre, x_fp32, input_ids, indices):
    """Run only requested terminal FFN rows; all attention/cache work stays full.

    This is the conservative V4.1 specialization: late DSpark features prevent
    broad YoCo layer compaction, but no consumer needs unrequested final FFN
    rows. The caller disables this path for prompt scoring and full HC output.
    """
    import copy

    from vllm.forward_context import get_forward_context, override_forward_context

    from vllm_ascend.ascend_forward_context import MoECommType
    from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method

    if not getattr(mlp, "_terminal_compaction_seen", False):
        from vllm.logger import logger

        logger.info("Terminal consumer FFN compaction: full_rows=%d requested_rows=%d", x.shape[0], indices.shape[0])
        mlp._terminal_compaction_seen = True
    indices = indices.long()
    selected = [tensor.index_select(0, indices) for tensor in (x, residual, post, comb, pre, x_fp32)]
    x, residual, post, comb, pre, x_fp32 = selected
    input_ids = input_ids.index_select(0, indices) if input_ids is not None else None
    outer = get_forward_context()
    inner = copy.copy(outer)
    inner.additional_kwargs = dict(outer.additional_kwargs)
    count = indices.shape[0]
    # Use the existing qualified small-batch AllGather implementation with the
    # target's expert layout; no communication state from the drafter is reused.
    owner = outer.additional_kwargs["moe_comm_method"].moe_config
    comm = get_moe_comm_method(MoECommType.ALLGATHER, owner)
    if comm is None:
        raise RuntimeError("Terminal compaction has no target-owned AllGather dispatcher")
    inner.additional_kwargs.update(
        moe_comm_type=MoECommType.ALLGATHER,
        moe_comm_method=comm,
        num_tokens=count,
        max_tokens_across_dp=count,
        max_tokens_across_pcp=count,
        padded_num_tokens=count,
        padded_length=count,
        num_tokens_across_dp=None,
        use_mega_moe=False,
    )
    inner.is_padding = outer.is_padding.index_select(0, indices) if outer.is_padding is not None else None
    inner.skip_compiled = True
    with override_forward_context(inner):
        x = mlp(x, input_ids=input_ids, hidden_states_fp32=x_fp32, already_sequence_parallel=False)
    outer.moe_layer_index = inner.moe_layer_index
    return x, residual, post, comb, pre


@torch.library.custom_op("v41_terminal_consumer::run", mutates_args=())
def run_terminal_moe(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    x_fp32: torch.Tensor,
    input_ids: torch.Tensor | None,
    indices: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from vllm.forward_context import get_forward_context

    mlp = get_forward_context().no_compile_layers[layer_name]
    return compact_terminal_moe(mlp, x, residual, post, comb, pre, x_fp32, input_ids, indices)


@run_terminal_moe.register_fake
def _fake_terminal_moe(x, residual, post, comb, pre, x_fp32, input_ids, indices, layer_name):
    rows = indices.shape[0]
    return (
        x.new_empty((rows, x.shape[-1])),
        residual.new_empty((rows, *residual.shape[1:])),
        post.new_empty((rows, *post.shape[1:])),
        comb.new_empty((rows, *comb.shape[1:])),
        pre.new_empty((rows, *pre.shape[1:])),
    )


@torch.library.custom_op("v41_terminal_consumer::prepare", mutates_args=())
def prepare_terminal_consumer(
    hidden: torch.Tensor,
    mix: torch.Tensor,
    positions: torch.Tensor,
    indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    window: int,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Publish consumer-local metadata without changing any producer buffers."""
    from dataclasses import replace

    from vllm.forward_context import get_forward_context

    context = get_forward_context()
    attn = context.no_compile_layers[layer_name]
    impl = attn.v41_impl
    if impl.role.is_kv_source or impl.role.is_index_source or impl.role.compress_ratio != 1:
        raise RuntimeError("Terminal token compaction requires a C1 KV/index consumer")
    if getattr(attn, "packed_cache_ops", None) is not None:
        raise RuntimeError("Terminal consumer compaction is not qualified for packed A5 caches")
    indices = indices.long()
    compact_positions = positions.index_select(0, indices)
    rows = indices.shape[0]
    originals = {}
    for prefix in (impl.swa_prefix, impl.long_kv_source_prefix, impl.index_k_source_prefix):
        if prefix is None or prefix in originals:
            continue
        old = context.attn_metadata[prefix]
        originals[prefix] = old
        updates = dict(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=None,
            positions=compact_positions,
            num_actual_tokens=rows,
            num_input_tokens=rows,
            num_prefill_tokens=rows,
            num_decode_tokens=0,
            num_decodes=0,
            max_query_len=window,
            slot_mapping=old.slot_mapping.index_select(0, indices),
            flat_slot_mapping=(
                old.flat_slot_mapping.index_select(0, indices) if old.flat_slot_mapping is not None else None
            ),
        )
        if prefix == impl.swa_prefix:
            name = attn.rotary_emb.layername
            # RoPE metadata is a RopeDataProxy, whose iteration slices token
            # rows rather than yielding mapping keys. This private SWA leaf
            # is consumed by this one terminal rotary layer only.
            cos = {name: old.cos[name].index_select(0, indices)}
            sin = {name: old.sin[name].index_select(0, indices)}
            # Bound unused early queries to initialized retained local KV.
            # The last query sees its full W entries, exactly as in the baseline.
            visible = (torch.arange(rows, device=positions.device, dtype=torch.int32) % window + 1).unsqueeze(1)
            updates.update(cos=cos, sin=sin, ori_topk_length=visible, consumer_query_rows=indices)
        context.attn_metadata[prefix] = replace(old, **updates)
    metadata = impl._get_layer_metadata(context.attn_metadata)
    long = metadata.attention
    swa = metadata.swa
    long.smla_metadata = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        attn.n_local_heads,
        1,
        attn.head_dim,
        cu_seqlens_q=query_start_loc.int(),
        seqused_ori_kv=swa.seq_lens,
        seqused_cmp_kv=long.cache_seq_lens,
        cmp_residual_kv=long.cmp_residual,
        batch_size=swa.num_reqs,
        max_seqlen_q=window,
        max_seqlen_ori_kv=swa.max_seq_len,
        max_seqlen_cmp_kv=long.max_cache_seq_len,
        ori_topk=0,
        ori_topk_length=swa.ori_topk_length,
        cmp_topk=impl.topology.index_topk,
        cmp_ratio=1,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=window - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
        has_ori_kv=True,
        has_cmp_kv=True,
    )
    # Metadata was emitted on the current stream, not the target's deferred
    # producer stream. The attention implementation recognizes this local scope.
    context.additional_kwargs["v41_terminal_original_metadata"] = originals
    if not getattr(attn, "_terminal_token_compaction_seen", False):
        from vllm.logger import logger

        logger.info("Terminal consumer token compaction: full_rows=%d retained_rows=%d", hidden.shape[0], rows)
        attn._terminal_token_compaction_seen = True
    return hidden.index_select(0, indices), mix.index_select(0, indices), compact_positions


@prepare_terminal_consumer.register_fake
def _fake_prepare_consumer(hidden, mix, positions, indices, query_start_loc, window, layer_name):
    rows = indices.shape[0]
    return (
        hidden.new_empty((rows, *hidden.shape[1:])),
        mix.new_empty((rows, *mix.shape[1:])),
        positions.new_empty((rows,)),
    )


@torch.library.custom_op("v41_terminal_consumer::restore", mutates_args=())
def restore_terminal_consumer(value: torch.Tensor) -> torch.Tensor:
    from vllm.forward_context import get_forward_context

    context = get_forward_context()
    originals = context.additional_kwargs.pop("v41_terminal_original_metadata", {})
    for prefix, original in originals.items():
        context.attn_metadata[prefix] = original
    return value.clone()


@restore_terminal_consumer.register_fake
def _fake_restore_consumer(value):
    return torch.empty_like(value)
