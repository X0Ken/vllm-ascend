# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the actual opaque FFN restoration with non-owning SP padding."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.models.deepseek_v41 import consumer_compaction_runtime as runtime


def test_sp_non_owner_padding_does_not_overwrite_real_row_zero():
    class MLP:
        is_sequence_parallel = True
        experts = SimpleNamespace(_prepare_router_input=lambda x, fp: fp, _apply_router_linear=lambda x: x)
        prefill_compaction_stats = {"executed_steps": 0}
        prefill_compaction_plan = (
            torch.tensor([0, 0, 3, 3]),
            2,
            True,
            torch.tensor([True, False, True, False]),
            torch.tensor([True, False, True, False]),
        )

        def __call__(self, x, **kwargs):
            assert kwargs["already_sequence_parallel"]
            context.moe_layer_index += 1
            return x * 2

    mlp = MLP()
    context = SimpleNamespace(
        no_compile_layers={"last": mlp}, additional_kwargs={"moe_comm_type": MoECommType.ALLTOALL}, moe_layer_index=39
    )
    x = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    with (
        patch("vllm.forward_context.get_forward_context", return_value=context),
        patch.object(runtime, "_small_terminal_mlp", side_effect=lambda module, rows, *args, **kwargs: rows * 2),
    ):
        result = runtime.run_terminal_ffn(x, x, None, "last")
    expected = torch.zeros_like(x)
    expected[[0, 3]] = x[[0, 3]] * 2
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert context.moe_layer_index == 39
    assert mlp.prefill_compaction_stats["verified_rows"] == 4
    assert mlp.prefill_compaction_stats["verified_real_rows"] == 2
    assert mlp.prefill_compaction_stats["max_ffn_abs_difference"] == 0


def test_full_fallback_preserves_sequence_parallel_ownership():
    class MLP:
        is_sequence_parallel = True
        prefill_compaction_plan = None

        def __call__(self, x, **kwargs):
            assert kwargs["already_sequence_parallel"]
            return x

    context = SimpleNamespace(no_compile_layers={"last": MLP()}, additional_kwargs={})
    x = torch.ones(8, 4)
    with patch("vllm.forward_context.get_forward_context", return_value=context):
        result = runtime.run_terminal_ffn(x, x, None, "last")
    torch.testing.assert_close(result, x, rtol=0, atol=0)
    assert result.data_ptr() != x.data_ptr()


def test_wrapped_sequence_parallel_constructor_and_owner_selection():
    import numpy as np
    from vllm.config import CUDAGraphMode

    from vllm_ascend.models.deepseek_v41 import prefill_compaction as policy

    parallel = SimpleNamespace(
        tensor_parallel_size=8,
        data_parallel_size=2,
        enable_expert_parallel=True,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        use_sequence_parallel_moe=True,
        enable_eplb=False,
    )
    config = SimpleNamespace(
        parallel_config=parallel, kv_transfer_config=None, cache_config=SimpleNamespace(swa_bounded_replay=False)
    )
    role = SimpleNamespace(compress_ratio=1, is_kv_source=False, is_index_source=False)
    runner = SimpleNamespace(
        is_internal_router=True,
        ascend_shared_experts=SimpleNamespace(),
        routed_input_transform=None,
        routed_output_transform=None,
    )
    layer = SimpleNamespace(
        mlp=SimpleNamespace(experts=runner), self_attn=SimpleNamespace(role=role, packed_cache_ops=None)
    )
    backbone = SimpleNamespace(_mtp_hidden_buffer=None, aux_hidden_state_layers=(37, 38, 39), layers=[layer])
    model = SimpleNamespace(language_model=SimpleNamespace(model=backbone))
    state = SimpleNamespace(
        model=model,
        vllm_config=config,
        device="cpu",
        prompt_logprobs_worker=SimpleNamespace(token_id_scores={}, uses_prompt_logprobs=np.array([False])),
    )
    batch = SimpleNamespace(
        idx_mapping_np=np.array([0]),
        num_reqs=1,
        logits_indices=torch.tensor([5119]),
        has_prefill=True,
        num_tokens=5120,
        num_tokens_after_padding=5120,
        num_computed_prefill_tokens_np=np.array([0]),
        prefill_len_np=np.array([5120]),
        cu_num_logits_np=np.array([0, 1]),
        num_scheduled_tokens=np.array([5120]),
    )
    dp = SimpleNamespace(world_size=2, rank_in_group=0, cpu_group=None)

    def census(tensor, **kwargs):
        tensor[:, 1].copy_(tensor[:, 0])

    with (
        patch.object(policy, "should_skip_allreduce_across_dp_group", return_value=False),
        patch.object(policy, "get_tensor_model_parallel_rank", return_value=7),
        patch.object(policy, "get_dp_group", return_value=dp),
        patch.object(policy, "get_mrv2_in_profile_run", return_value=False),
        patch.object(policy.dist, "all_reduce", side_effect=census),
    ):
        helper = policy.TerminalPrefillCompaction(state)
        helper.verify = True
        helper.prepare(batch, CUDAGraphMode.NONE, False)
    indices, count, verify, valid, used = layer.mlp.prefill_compaction_plan
    assert count == 1 and verify
    assert used.tolist() == valid.tolist()
    assert indices[0] == 639
    assert valid.tolist() == [True, False, False, False, False, False, False, False]
    helper.finish()
    assert layer.mlp.prefill_compaction_plan is None


def test_compact_context_counts_full_sp_gather_for_shared_experts():
    from contextlib import contextmanager

    captured = []

    @contextmanager
    def override(context):
        captured.append(context)
        yield context

    class MLP:
        is_sequence_parallel = True
        tp_size = 8

        def __call__(self, rows, **kwargs):
            assert kwargs["already_sequence_parallel"]
            return rows

    comm = SimpleNamespace(moe_config=object())
    context = SimpleNamespace(
        additional_kwargs={"moe_comm_type": MoECommType.ALLTOALL, "moe_comm_method": comm},
        is_padding=None,
        moe_layer_index=39,
    )
    x = torch.ones(8, 4)
    with (
        patch("vllm.forward_context.get_forward_context", return_value=context),
        patch("vllm.forward_context.override_forward_context", side_effect=override),
    ):
        runtime._small_terminal_mlp(MLP(), x, x, None, torch.arange(8), preserve_backend=True)
    assert captured[0].additional_kwargs["num_tokens"] == 64
    assert captured[0].additional_kwargs["padded_num_tokens"] == 64
    assert "num_tokens" not in context.additional_kwargs


def test_full_vocabulary_oracle_uses_real_head_and_only_consumed_rows():
    from vllm_ascend.models.deepseek_v41 import prefill_compaction as policy

    layer = SimpleNamespace(hc_post=lambda ffn, *args: ffn, hc_collapse=lambda hidden, pre: hidden)
    helper = object.__new__(policy.TerminalPrefillCompaction)
    helper.backbone = SimpleNamespace(layers=[layer], norm=lambda hidden: hidden)
    helper.state = SimpleNamespace(model=SimpleNamespace(compute_logits=lambda hidden: hidden))
    helper.stats = {}
    helper.sequence_parallel = True
    helper.oracle_indices = torch.tensor([0, 1])
    helper.oracle_used = torch.tensor([True, False])
    tp = SimpleNamespace(all_gather=lambda tensor, **kwargs: tensor)
    reference = torch.tensor([[3.0, 1.0], [4.0, 1.0]])
    compact = torch.tensor([[2.9, 1.0], [1.0, 4.0]])
    indices = torch.tensor([0, 1])
    used = torch.tensor([True, False])
    with patch.object(policy, "get_tp_group", return_value=tp):
        helper.verify_logits(reference, compact, indices, used, reference, reference, reference, reference)
        assert helper.stats["greedy_mismatches"] == 0
        compact[0] = torch.tensor([1.0, 3.0])
        helper.verify_logits(reference, compact, indices, used, reference, reference, reference, reference)
    assert helper.stats["greedy_mismatches"] == 1
    assert helper.stats["verified_logit_rows"] == 2


def test_sp_dense_decode_and_single_token_idle_peers_do_not_veto_prefill():
    import numpy as np
    from vllm.config import CUDAGraphMode

    from vllm_ascend.models.deepseek_v41 import prefill_compaction as policy

    helper = object.__new__(policy.TerminalPrefillCompaction)
    helper.state = SimpleNamespace(
        device="cpu", prompt_logprobs_worker=SimpleNamespace(token_id_scores={}, uses_prompt_logprobs=np.array([False]))
    )
    helper.sequence_parallel = True
    helper.tp_rank = 7
    helper.verify = False
    helper.mlp = SimpleNamespace()
    helper.stats = dict(admitted_steps=0, executed_steps=0, original_rows=0, requested_rows=0, padded_rows=0)
    batch = SimpleNamespace(
        is_dummy=False,
        idx_mapping_np=np.array([0]),
        num_reqs=1,
        logits_indices=torch.arange(96),
        has_prefill=False,
        num_tokens=96,
        num_tokens_after_padding=96,
    )
    dp = SimpleNamespace(world_size=2, rank_in_group=0, cpu_group=None)

    def census(tensor, **kwargs):
        tensor[:, 1] = torch.tensor([1, 1, 1, 1024, 0])

    with (
        patch.object(policy, "get_dp_group", return_value=dp),
        patch.object(policy, "get_mrv2_in_profile_run", return_value=False),
        patch.object(policy.dist, "all_reduce", side_effect=census),
    ):
        helper.prepare(batch, CUDAGraphMode.NONE, False)
        indices, _, _, valid, _ = helper.mlp.prefill_compaction_plan
        assert indices.tolist() == list(range(12))
        assert valid.all()
        batch.is_dummy = True
        batch.num_tokens = batch.num_tokens_after_padding = 1
        batch.logits_indices = torch.tensor([0])
        helper.prepare(batch, CUDAGraphMode.NONE, False)
        indices, _, _, valid, _ = helper.mlp.prefill_compaction_plan
        assert indices.tolist() == [0]
        assert not valid.any()


def test_terminal_router_uses_original_full_shape_logits_only_for_its_owner():
    from vllm_ascend.ops.fused_moe import fused_moe as moe

    logits = torch.ones(2, 4)
    owner = SimpleNamespace(
        v41_terminal_compaction_owner=True,
        is_internal_router=True,
        _prepare_router_input=lambda *args: (_ for _ in ()).throw(AssertionError("Router must retain its full GEMM")),
        _apply_router_linear=lambda *args: None,
        _prepare_shared_for_router_overlap=lambda *args: None,
        ascend_shared_experts=SimpleNamespace(),
    )
    context = SimpleNamespace(additional_kwargs={"v41_terminal_ffn_full": {"router_logits": logits}})
    with (
        patch.object(moe, "get_forward_context", return_value=context),
        patch.object(moe, "maybe_record_event", return_value=None),
    ):
        result = moe.AscendMoERunner._prepare_router_and_milestones(owner, None, torch.zeros(2, 4), torch.zeros(2, 4))
    assert result[0] is logits


def test_terminal_shared_shape_and_output_selection_preserve_dense_work():
    from contextlib import nullcontext

    from vllm_ascend.ops.fused_moe import shared_experts as shared
    from vllm_ascend.ops.fused_moe.dataclass.shared_experts import PreparedSharedExpertInput, RoutedMoEMilestones

    helper = object.__new__(shared.AscendSharedExperts)
    helper.v41_terminal_compaction_owner = True
    helper.multistream_overlap = False
    helper.parallel_mode = lambda: shared.SharedExpertParallelMode.TENSOR_PARALLEL
    full = torch.arange(20).reshape(5, 4).float()
    helper._finish_shared_mlp = lambda *args: full
    indices = torch.tensor([4, 1])
    context = SimpleNamespace(additional_kwargs={"v41_terminal_ffn_full": {"num_tokens": 40, "indices": indices}})
    with (
        patch.object(shared, "get_forward_context", return_value=context),
        patch.object(shared, "npu_stream_switch", side_effect=lambda *args, **kwargs: nullcontext()),
    ):
        assert helper._shared_num_tokens() == 40
        output = helper.forward(
            PreparedSharedExpertInput(full),
            RoutedMoEMilestones(),
            submitted_gate_up=SimpleNamespace(local_dp_metadata=None),
        )
    torch.testing.assert_close(output, full.index_select(0, indices), rtol=0, atol=0)


def test_oracle_also_checks_the_benchmark_ignore_eos_decision():
    from vllm_ascend.models.deepseek_v41 import prefill_compaction as policy

    helper = object.__new__(policy.TerminalPrefillCompaction)
    helper.backbone = SimpleNamespace(
        layers=[SimpleNamespace(hc_post=lambda ffn, *args: ffn, hc_collapse=lambda x, pre: x)], norm=lambda x: x
    )
    helper.state = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(eos_token_id=0), compute_logits=lambda x: x)
    )
    helper.stats = {}
    helper.sequence_parallel = True
    helper.oracle_indices = torch.tensor([0])
    helper.oracle_used = torch.tensor([True])
    reference = torch.tensor([[10.0, 3.0, 1.0]])
    compact = torch.tensor([[10.0, 1.0, 3.0]])
    with patch.object(policy, "get_tp_group", return_value=SimpleNamespace(all_gather=lambda x, **kwargs: x)):
        helper.verify_logits(reference, compact, None, None, reference, reference, reference, reference)
    assert helper.stats["greedy_mismatches"] == 0
    assert helper.stats["ignore_eos_greedy_mismatches"] == 1
