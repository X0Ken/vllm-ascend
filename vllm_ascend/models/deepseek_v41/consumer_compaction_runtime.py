# SPDX-License-Identifier: Apache-2.0
"""Conservative terminal-consumer MoE compaction outside captured graphs."""

import torch


def compact_terminal_moe(mlp, x, residual, post, comb, pre, x_fp32, input_ids, indices):
    """Run only requested terminal FFN rows; all attention/cache work stays full.

    This is the conservative V4.1 specialization: late DSpark features prevent
    broad YoCo layer compaction, but no consumer needs unrequested final FFN
    rows. The caller disables this path for prompt scoring and full HC output.
    """
    if not getattr(mlp, "_terminal_compaction_seen", False):
        from vllm.logger import logger

        logger.info("Terminal consumer FFN compaction: full_rows=%d requested_rows=%d", x.shape[0], indices.shape[0])
        mlp._terminal_compaction_seen = True
    indices = indices.long()
    selected = [tensor.index_select(0, indices) for tensor in (x, residual, post, comb, pre, x_fp32)]
    x, residual, post, comb, pre, x_fp32 = selected
    input_ids = input_ids.index_select(0, indices) if input_ids is not None else None
    x = _small_terminal_mlp(mlp, x, x_fp32, input_ids, indices)
    return x, residual, post, comb, pre


def _small_terminal_mlp(
    mlp, x, x_fp32, input_ids, indices, *, preserve_backend=False, shared_full=None, router_logits=None
):
    """Run compact routed experts within the target-owned communication context."""
    import copy

    from vllm.forward_context import get_forward_context, override_forward_context

    from vllm_ascend.ascend_forward_context import MoECommType
    from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method

    outer = get_forward_context()
    inner = copy.copy(outer)
    inner.additional_kwargs = dict(outer.additional_kwargs)
    count = x.shape[0]
    # Shared TP experts gather all SP shards before projecting. Their unpad
    # length must describe the gathered sequence, not one rank's shard.
    global_count = count * mlp.tp_size if mlp.is_sequence_parallel else count
    # Use the existing qualified small-batch AllGather implementation with the
    # target's expert layout; no communication state from the drafter is reused.
    owner = outer.additional_kwargs["moe_comm_method"].moe_config
    kind = outer.additional_kwargs["moe_comm_type"] if preserve_backend else MoECommType.ALLGATHER
    comm = outer.additional_kwargs["moe_comm_method"] if preserve_backend else get_moe_comm_method(kind, owner)
    if comm is None:
        raise RuntimeError("Terminal compaction has no target-owned AllGather dispatcher")
    inner.additional_kwargs.update(
        moe_comm_type=kind,
        moe_comm_method=comm,
        num_tokens=global_count,
        max_tokens_across_dp=global_count,
        max_tokens_across_pcp=global_count,
        padded_num_tokens=global_count,
        padded_length=global_count,
        num_tokens_across_dp=None,
        use_mega_moe=False,
    )
    if shared_full is not None:
        inner.additional_kwargs["v41_terminal_ffn_full"] = {
            "input": shared_full,
            "num_tokens": outer.additional_kwargs["num_tokens"],
            "indices": indices,
            "router_logits": router_logits,
        }
    inner.is_padding = outer.is_padding.index_select(0, indices) if outer.is_padding is not None else None
    inner.skip_compiled = True
    with override_forward_context(inner):
        x = mlp(x, input_ids=input_ids, hidden_states_fp32=x_fp32, already_sequence_parallel=mlp.is_sequence_parallel)
    outer.moe_layer_index = inner.moe_layer_index
    return x


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


@torch.library.custom_op("v41_terminal_consumer::ffn", mutates_args=())
def run_terminal_ffn(
    x: torch.Tensor,
    x_fp32: torch.Tensor,
    input_ids: torch.Tensor | None,
    layer_name: str,
    residual: torch.Tensor | None = None,
    post: torch.Tensor | None = None,
    comb: torch.Tensor | None = None,
    pre: torch.Tensor | None = None,
) -> torch.Tensor:
    """Preserve all attention/KV/DSpark rows and prune only final FFN work."""
    from vllm.forward_context import get_forward_context

    mlp = get_forward_context().no_compile_layers[layer_name]
    plan = getattr(mlp, "prefill_compaction_plan", None)
    from vllm_ascend.ascend_forward_context import MoECommType

    kind = get_forward_context().additional_kwargs.get("moe_comm_type")
    if (
        plan is None
        or (mlp.is_sequence_parallel and kind == MoECommType.ALLGATHER)
        or kind
        not in (
            MoECommType.ALLTOALL,
            MoECommType.ALLGATHER,
        )
    ):
        return mlp(
            x, input_ids=input_ids, hidden_states_fp32=x_fp32, already_sequence_parallel=mlp.is_sequence_parallel
        ).clone()
    indices, real_count, verify, valid, used = plan
    reference = None
    if verify:
        context = get_forward_context()
        layer_index = context.moe_layer_index
        reference_full = mlp(
            x, input_ids=input_ids, hidden_states_fp32=x_fp32, already_sequence_parallel=mlp.is_sequence_parallel
        )
        reference_full = reference_full.clone()
        reference = reference_full.index_select(0, indices)
        # The oracle is an extra call, not an extra logical layer. Both runs
        # must resolve the same target MoE weights through the layer cursor.
        context.moe_layer_index = layer_index
        replay_full = mlp(
            x, input_ids=input_ids, hidden_states_fp32=x_fp32, already_sequence_parallel=mlp.is_sequence_parallel
        )
        replay_full = replay_full.clone()
        replay = replay_full.index_select(0, indices)
        context.moe_layer_index = layer_index
    selected = x.index_select(0, indices)
    fp32 = x_fp32.index_select(0, indices)
    ids = input_ids.index_select(0, indices) if input_ids is not None else None
    # Keep the original Router GEMM shape and shared TP reduction. Only the
    # routed experts dispatch/GMM work changes row count.
    runner = mlp.experts
    router_input = runner._prepare_router_input(x, x_fp32)
    full_router_logits = runner._apply_router_linear(router_input)
    compact = _small_terminal_mlp(
        mlp,
        selected,
        fp32,
        ids,
        indices,
        preserve_backend=True,
        shared_full=x,
        router_logits=full_router_logits.index_select(0, indices),
    )
    stats = mlp.prefill_compaction_stats
    stats["executed_steps"] += 1
    if reference is not None:
        # Explicit diagnostic mode only: device reads never enter performance waves.
        self_error = torch.where(valid.unsqueeze(1), (reference.float() - replay.float()).abs(), 0).max().item()
        stats["max_full_ffn_self_abs_difference"] = max(stats.get("max_full_ffn_self_abs_difference", 0.0), self_error)
        error = (reference.float() - compact.float()).abs()
        maximum = torch.where(valid.unsqueeze(1), error, 0).max().item()
        stats["max_padding_ffn_abs_difference"] = max(
            stats.get("max_padding_ffn_abs_difference", 0.0), error.max().item()
        )
        stats["verified_steps"] = stats.get("verified_steps", 0) + 1
        stats["verified_rows"] = stats.get("verified_rows", 0) + indices.numel()
        stats["verified_real_rows"] = stats.get("verified_real_rows", 0) + valid.sum().item()
        stats["max_ffn_abs_difference"] = max(stats.get("max_ffn_abs_difference", 0.0), maximum)
    result = torch.zeros_like(x)
    # Padding may repeat row zero; only unique real rows are ever restored.
    result.index_add_(0, indices, torch.where(valid.unsqueeze(1), compact, 0))
    if verify and residual is not None:
        mlp.prefill_compaction_oracle(
            reference_full, result, indices, used, residual, post, comb, pre, control=replay_full
        )
    return result


@run_terminal_ffn.register_fake
def _fake_terminal_ffn(x, x_fp32, input_ids, layer_name, residual=None, post=None, comb=None, pre=None):
    return torch.empty_like(x)
