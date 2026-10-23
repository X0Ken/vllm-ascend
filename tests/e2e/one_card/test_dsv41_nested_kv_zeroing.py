# SPDX-License-Identifier: Apache-2.0
"""Strict NPU block-zeroing coverage for nested MRV2 forward bindings."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.worker.v2.utils import AscendV2KVBlockZeroer


@pytest.mark.parametrize("cache_dtype", ["auto", "mxfp8"])
@pytest.mark.parametrize("ratio", [1, 2])
def test_nested_cache_zeroing_preserves_other_blocks_and_static_scale(cache_dtype, ratio):
    blocks = 4
    tensors = [torch.full((blocks * ratio, 16, 1, 8), float(i + 3), device="npu") for i in range(4)]
    dynamic = tensors[:3]
    binding = [tuple(tensors)] if cache_dtype == "mxfp8" else [(dynamic[0], (dynamic[1], [dynamic[2], torch.empty(0)]))]
    spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float32)
    group = AttentionGroup(AscendAttentionBackend, ["layer"], spec, 0)
    zeroer = AscendV2KVBlockZeroer(
        torch.device("npu"),
        [group],
        [16],
        {"layer": SimpleNamespace(kv_cache=binding)},
        blocks,
        cache_dtype=cache_dtype,
    )
    assert len(zeroer._zeroers) == 3
    zeroer.zero_block_ids([1, 3])
    torch.npu.synchronize()
    for index, actual in enumerate(dynamic):
        expected = torch.full(actual.shape, float(index + 3))
        for block_id in [1, 3]:
            expected[block_id * ratio : (block_id + 1) * ratio].zero_()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(tensors[3].cpu(), torch.full(tensors[3].shape, 6.0), rtol=0, atol=0)
