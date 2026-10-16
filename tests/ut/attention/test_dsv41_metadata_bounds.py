# SPDX-License-Identifier: Apache-2.0
"""Host tiling bounds must never replace exact device attention lengths."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionState, AscendCommonAttentionMetadata
from vllm_ascend.attention.dsa_v41 import AscendDSAV41MetadataBuilder
from vllm_ascend.spec_decode import llm_base_proposer
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


@pytest.mark.parametrize("extra", [0, 1, 5, 128])
@pytest.mark.parametrize("prefill", [False, True])
def test_bounds_preserve_device_lengths_and_request_classification(extra, prefill):
    builder = AscendDSAV41MetadataBuilder.__new__(AscendDSAV41MetadataBuilder)
    builder._seq_lens = torch.zeros(4, dtype=torch.int32)
    actual = torch.tensor([127, 256], dtype=torch.int32)
    qsl = torch.tensor([0, 5, 10], dtype=torch.int32)
    common = SimpleNamespace(
        seq_lens=actual,
        seq_lens_cpu=None,
        _seq_lens_cpu=None,
        seq_lens_cpu_upper_bound=actual + extra,
        max_seq_len=256 + extra,
        positions=torch.arange(10),
        query_start_loc=qsl,
        query_start_loc_cpu=qsl,
        is_prefilling=torch.tensor([prefill, False]),
    )
    bounded = builder._build_batch_metadata(common, 2, 2, 10)
    torch.testing.assert_close(bounded["seq_lens"], actual)
    assert bounded["seq_lens_cpu"] is None
    assert bounded["max_cache_seq_len"] == 256 + extra
    common.seq_lens_cpu = actual
    reference = builder._build_batch_metadata(common, 2, 2, 10)
    for key in ["seq_lens", "positions", "query_start_loc"]:
        torch.testing.assert_close(bounded[key], reference[key])
    for key in ["num_decodes", "num_prefills", "num_decode_tokens", "num_prefill_tokens"]:
        assert bounded[key] == reference[key]


@pytest.mark.parametrize("exact_host", [False, True])
def test_padded_draft_keeps_upper_bound_when_exact_host_is_unset(exact_host):
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.dcp_size = 1
    proposer.arange = torch.arange(16, dtype=torch.int32)
    proposer.runner = MagicMock()
    proposer.runner.dcp_manager = None
    proposer.runner.actual_seq_lengths_q = [3, 3]
    proposer.runner.attn_state = AscendAttentionState.SpecDecoding
    proposer.runner.decode_token_per_req = 4

    internal_seq_lens_cpu = torch.tensor([7, 9], dtype=torch.int32) if exact_host else None
    upper_bound = torch.tensor([12, 14], dtype=torch.int32)
    common_attn_metadata = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 3, 6], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 3, 6], dtype=torch.int32),
        seq_lens=torch.tensor([7, 9], dtype=torch.int32),
        _seq_lens_cpu=internal_seq_lens_cpu,
        seq_lens_cpu_upper_bound=upper_bound,
        seq_lens_cpu=None,
        num_computed_tokens_cpu=None,
        num_reqs=2,
        num_actual_tokens=6,
        num_input_tokens=6,
        max_query_len=3,
        actual_seq_lengths_q=[3, 3],
        block_table_tensor=torch.zeros((2, 1), dtype=torch.int32),
        slot_mapping=torch.arange(6, dtype=torch.int32),
        positions=torch.arange(6),
        attn_state=AscendAttentionState.SpecDecoding,
        decode_token_per_req=4,
        max_seq_len=14,
    )
    spec_decode_metadata = MagicMock()
    spec_decode_metadata.cu_num_draft_tokens = torch.tensor([2, 3], dtype=torch.int32)
    valid_sampled_tokens_count = torch.tensor([3, 1], dtype=torch.int32)

    with patch.object(llm_base_proposer, "HAS_TRITON", False):
        spec_common_attn_metadata, *_ = proposer.prepare_inputs_padded(
            common_attn_metadata,
            spec_decode_metadata,
            valid_sampled_tokens_count,
        )

    assert spec_common_attn_metadata._seq_lens_cpu is internal_seq_lens_cpu
    assert spec_common_attn_metadata.seq_lens_cpu is None

    assert spec_common_attn_metadata.max_seq_len == 14
    assert spec_common_attn_metadata.seq_lens_cpu_upper_bound is upper_bound
