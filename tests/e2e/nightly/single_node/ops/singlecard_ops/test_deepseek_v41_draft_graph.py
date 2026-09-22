# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay draft SWA with changed lengths, physical blocks, queries and padding."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention import dsa_v41
from vllm_ascend.core.deepseek_v41 import DeepseekV41DraftSWASpec
from vllm_ascend.ops import rope_dsv4
from vllm_ascend.utils import enable_custom_op


@pytest.mark.parametrize("batch", [1, 3, 8, 32])
@pytest.mark.parametrize("query_len", [1, 5])
@torch.inference_mode()
def test_draft_swa_graph_replays_changed_metadata(monkeypatch, batch, query_len):
    assert enable_custom_op()
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    torch.manual_seed(143)
    flags = SimpleNamespace(enable_dsv41_draft_graph=False, enable_dsv41_metadata_fusion=False)
    monkeypatch.setattr(dsa_v41, "get_ascend_config", lambda: flags)
    monkeypatch.setattr(rope_dsv4, "_ROPE_STATE", rope_dsv4.RopeGlobalState())
    runtime = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=dict(
                sliding_window=128,
                num_attention_heads=32,
                head_dim=512,
                qk_rope_head_dim=64,
                index_topk=512,
                index_n_heads=64,
                index_head_dim=128,
            )
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=512, max_num_seqs=32),
        speculative_config=SimpleNamespace(num_speculative_tokens=query_len, use_eagle=lambda: True),
    )
    layer = "model.layers.0.self_attn.attn"
    with device:
        rope_dsv4.ComplexExpRotaryEmbedding(
            runtime,
            layer,
            head_size=512,
            rotary_dim=64,
            max_position_embeddings=1024,
            base=10000,
            scaling_factor=1,
        )
    spec = DeepseekV41DraftSWASpec(
        block_size=128,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=128,
        cache_dtype_str="bfloat16",
        model_version="deepseek_v4",
    )
    reference = dsa_v41.DeepseekV41MetadataBuilder(spec, [layer], runtime, device)
    candidate = dsa_v41.DeepseekV41MetadataBuilder(spec, [layer], runtime, device)
    cache = torch.randn(batch * 8 + 8, 128, 1, 512, dtype=torch.bfloat16, device=device)
    query = torch.empty(batch * query_len, 4, 512, dtype=torch.bfloat16, device=device)
    attention = SimpleNamespace(
        head_dim=512,
        window_size=128,
        n_local_heads=4,
        softmax_scale=512**-0.5,
        attn_sink=torch.zeros(4, dtype=torch.float32, device=device),
        dsa_attn=SimpleNamespace(swa_cache_layer=SimpleNamespace(kv_cache=[cache])),
    )
    impl = dsa_v41.DeepseekV41EagerAttentionImpl.__new__(dsa_v41.DeepseekV41EagerAttentionImpl)
    impl.role = SimpleNamespace(compress_ratio=0)
    impl.topology = SimpleNamespace(index_topk=512)

    def consume(entry):
        q = query.clone()
        cos, sin = entry.cos[layer], entry.sin[layer]
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1), cos, sin, rotary_mode="interleave", partial_slice=[448, 512]
        )
        return impl._native_attention(
            attention,
            q,
            SimpleNamespace(swa=entry, attention=None),
            source_cache=None,
            compressed_indices=None,
        )

    fields = (
        "query_start_loc",
        "seq_lens",
        "block_table",
        "slot_mapping",
        "ori_sparse_indices",
        "ori_topk_length",
        "smla_metadata",
    )
    graph = None
    pointers = None
    previous = None
    for iteration, active in enumerate([batch, max(1, batch - 1), 0, batch]):
        # New input allocations make accidental capture of temporary pointers visible.
        lengths = [
            ([127, 128, 255, 512][(row + iteration) % 4] + query_len)
            if row < active
            else (query_len if active == 0 else 0)
            for row in range(batch)
        ]
        table_cpu = torch.arange(1, batch * 8 + 1, dtype=torch.int32).reshape(batch, 8) + iteration
        positions_cpu = torch.tensor(
            [
                pos
                for length in lengths
                for pos in (range(length - query_len, length) if length and active else [0] * query_len)
            ],
            dtype=torch.int64,
        )
        slots = torch.tensor(
            [
                int(table_cpu[row, pos // 128]) * 128 + pos % 128 if row < active else -1
                for row in range(batch)
                for pos in positions_cpu[row * query_len : (row + 1) * query_len]
            ],
            dtype=torch.int64,
            device=device,
        )
        qsl_cpu = torch.arange(batch + 1, dtype=torch.int32) * query_len
        common = dict(
            query_start_loc=qsl_cpu.to(device),
            query_start_loc_cpu=qsl_cpu,
            seq_lens=torch.tensor(lengths, dtype=torch.int32, device=device),
            seq_lens_cpu=torch.tensor(lengths, dtype=torch.int32),
            positions=positions_cpu.to(device),
            slot_mapping=slots,
            block_table_tensor=table_cpu.to(device),
            num_reqs=batch,
            num_input_tokens=batch * query_len,
            num_actual_tokens=batch * query_len,
            max_query_len=query_len,
            max_seq_len=max(lengths),
            is_prefilling=torch.ones(batch, dtype=torch.bool),
            causal=False,
        )
        query.normal_()
        flags.enable_dsv41_draft_graph = False
        expected_meta = reference.build_for_drafting(SimpleNamespace(**common), 1)
        expected = consume(expected_meta).cpu()
        flags.enable_dsv41_draft_graph = True
        actual_meta = candidate.build_for_drafting(SimpleNamespace(**common), 1)
        for field in fields:
            expected_tensor, actual_tensor = getattr(expected_meta, field), getattr(actual_meta, field)
            # Only the initialized native metadata payload is compared.
            if field == "smla_metadata":
                expected_tensor, actual_tensor = expected_tensor[:900], actual_tensor[:900]
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
        current = tuple(getattr(actual_meta, field).data_ptr() for field in fields)
        current += (actual_meta.cos[layer].data_ptr(), actual_meta.sin[layer].data_ptr())
        if pointers is not None:
            assert current == pointers
        pointers = current
        if graph is None:
            # Warm native operators before capture.
            consume(actual_meta)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
                output = consume(actual_meta)
        graph.replay()
        torch.npu.synchronize()
        actual = output.cpu()
        # SMLA leaves seq_len=0 padding outputs unspecified. The proposer
        # discards those rows; compare every live query, plus all rows of the
        # nonempty dummy batch used for the fully idle DP rank.
        checked_tokens = (active if active else batch) * query_len
        torch.testing.assert_close(actual[:checked_tokens], expected[:checked_tokens], rtol=0, atol=0)
        assert torch.isfinite(actual[:checked_tokens]).all()
        if previous is not None and active:
            assert not torch.equal(previous, actual)
        previous = actual
