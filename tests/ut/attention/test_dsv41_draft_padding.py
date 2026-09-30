# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention import deepseek_v41_slots, dsa_v41
from vllm_ascend.core.deepseek_v41 import DeepseekV41DraftSWASpec


@pytest.mark.parametrize("query_len", [1, 5])
@pytest.mark.parametrize("input_padding", [0, 5])
def test_draft_indices_follow_attention_queries(monkeypatch, query_len, input_padding):
    # Slot conversion is an NPU-only Triton kernel, covered by the native test.
    monkeypatch.setattr(
        deepseek_v41_slots,
        "slot_coordinates",
        lambda builder, common, positions, n, *args: builder._slot_mapping_2d[:n].fill_(-1),
    )
    monkeypatch.setattr(dsa_v41, "get_ascend_config", lambda: SimpleNamespace(enable_dsv41_draft_graph=True))
    runtime = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=dict(sliding_window=128, num_attention_heads=32, head_dim=512, index_topk=512)
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64, max_num_seqs=8),
        speculative_config=SimpleNamespace(num_speculative_tokens=query_len),
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
    builder = dsa_v41.DeepseekV41MetadataBuilder(spec, [], runtime, torch.device("cpu"))
    pointers = None
    # Populate a large batch first, then shrink/grow across both reported
    # failures (15 -> 20 and 30 -> 35) and an empty local batch.
    for batch in [8, 3, 0, 6, 7, 8]:
        tokens = batch * query_len
        qsl = torch.arange(batch + 1, dtype=torch.int32) * query_len
        lengths = torch.tensor([127 + 133 * (row % 2) for row in range(batch)], dtype=torch.int32)
        common = SimpleNamespace(
            num_reqs=batch,
            num_actual_tokens=tokens,
            num_input_tokens=tokens + input_padding,
            query_start_loc=qsl,
            query_start_loc_cpu=qsl,
            seq_lens=lengths,
            seq_lens_cpu=lengths,
            block_table_tensor=torch.ones(batch, 3, dtype=torch.int32),
            slot_mapping=torch.full((tokens + input_padding,), -1, dtype=torch.int64),
            positions=None,
            max_query_len=query_len,
            max_seq_len=260,
            causal=False,
        )
        metadata = builder.build_for_drafting(common, 1)
        assert metadata.num_input_tokens == tokens + input_padding
        assert metadata.num_actual_tokens == tokens
        assert metadata.ori_sparse_indices.shape == (tokens, 1, 256)
        assert metadata.ori_topk_length.shape == (tokens, 1)
        for row, length in enumerate(lengths.tolist()):
            visible = list(range(max(0, length - query_len - 128), length))
            expected = torch.tensor(visible + [-1] * (256 - len(visible)), dtype=torch.int32)
            for token in range(row * query_len, (row + 1) * query_len):
                torch.testing.assert_close(metadata.ori_sparse_indices[token, 0], expected)
                assert metadata.ori_topk_length[token, 0] == len(visible)
        current = (builder._draft_sparse_indices.data_ptr(), builder._draft_topk_length.data_ptr())
        if pointers is not None:
            assert current == pointers
        pointers = current
        if tokens:
            assert metadata.ori_sparse_indices.data_ptr() == current[0]
            assert metadata.ori_topk_length.data_ptr() == current[1]
