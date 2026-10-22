# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch_npu  # noqa: F401


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_allgather_dispatch_mask_fusion_is_bitwise_exact_for_strided_views(dtype):
    from vllm_ascend.ops.triton.moe_dispatch_mask import mask_expert_weights

    torch.manual_seed(193)
    weights = torch.randn(96, 16, device="npu", dtype=dtype)[:, ::2]
    expert_ids = torch.randint(-256, 256, (96, 16), device="npu", dtype=torch.int32)[:, ::2]
    expert_map = torch.full((512,), -1, device="npu", dtype=torch.int32)[::2]
    expert_map[32:64] = torch.arange(32, device="npu", dtype=torch.int32)
    expected = weights * (expert_map[expert_ids.long()] != -1)
    actual = mask_expert_weights(weights, expert_ids, expert_map)
    assert torch.equal(actual.view(torch.uint8), expected.contiguous().view(torch.uint8))
