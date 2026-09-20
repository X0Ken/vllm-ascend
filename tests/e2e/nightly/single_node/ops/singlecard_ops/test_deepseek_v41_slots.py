# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare fused slot coordinates with the original tensor implementation."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.deepseek_v41_slots import _slots, slot_coordinates


@pytest.mark.parametrize("n", [0, 1, 5, 6, 24, 129, 512])
@pytest.mark.parametrize("ratio,compressed", [(1, False), (1, True), (2, True)])
@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("storage", [16, 32, 128])
@pytest.mark.parametrize("has_positions", [False, True])
def test_slot_coordinates(n, ratio, compressed, skip, storage, has_positions):
    torch.manual_seed(17)
    raw = torch.randint(-4, 40000, (n,), dtype=torch.int64, device="npu")
    positions = torch.arange(n, device="npu", dtype=torch.int64) + 31999 if has_positions else None
    common = SimpleNamespace(
        slot_mapping=raw,
        query_start_loc=torch.tensor([0, max(0, n - 2)], device="npu", dtype=torch.int32),
    )
    # Keep a padded tail to detect writes outside the requested token span.
    buffer = torch.full((n + 3, 2), -99, device="npu", dtype=torch.int32)
    builder = SimpleNamespace(_slot_mapping_2d=buffer, kv_cache_spec=SimpleNamespace(storage_block_size=storage))
    got = slot_coordinates(builder, common, positions, n, 1, max(0, n - 1), ratio, compressed, skip)
    active = raw
    if compressed and ratio != 1:
        active = torch.where((raw >= 0) & ((raw + 1) % ratio == 0), raw // ratio, -1)
    valid = active >= 0
    if compressed and ratio == 2:
        if skip:
            valid.zero_()
        else:
            valid &= torch.arange(n, device="npu") < min(max(0, n - 2), max(0, n - 1))
            if positions is not None:
                valid &= positions % 2 == 1
    physical = active.clamp_min(0)
    expected = torch.stack(
        [torch.where(valid, physical // storage, -1), torch.where(valid, physical % storage, -1)], dim=1
    ).int()
    assert torch.equal(got, expected)
    assert torch.all(buffer[n:] == -99)
    if n:
        assert got.data_ptr() == buffer.data_ptr()


def test_dynamic_lengths_reuse_compiled_kernel():
    raw = torch.arange(4096, device="npu", dtype=torch.int64) - 3
    positions = torch.arange(4096, device="npu", dtype=torch.int64)
    common = SimpleNamespace(
        slot_mapping=raw,
        query_start_loc=torch.arange(17, device="npu", dtype=torch.int32) * 257,
    )
    buffer = torch.full((4099, 2), -99, device="npu", dtype=torch.int32)
    builder = SimpleNamespace(_slot_mapping_2d=buffer, kv_cache_spec=SimpleNamespace(storage_block_size=128))
    cases = [(4096, 16, 4095), (1, 0, 0), (1, 1, 1), (5, 2, 3), (129, 8, 127), (512, 16, 511), (4096, 3, 1025)]
    compiled_variants = None
    for n, actual_reqs, actual_tokens in cases:
        buffer.fill_(-99)
        got = slot_coordinates(builder, common, positions, n, actual_reqs, actual_tokens, 2, True, False)
        active = raw[:n]
        valid = (active >= 0) & ((active + 1) % 2 == 0)
        valid &= torch.arange(n, device="npu") < min(actual_reqs * 257, actual_tokens)
        valid &= positions[:n] % 2 == 1
        physical = (active // 2).clamp_min(0)
        expected = torch.stack(
            [torch.where(valid, physical // 128, -1), torch.where(valid, physical % 128, -1)], dim=1
        ).int()
        assert torch.equal(got, expected)
        assert torch.all(buffer[n:] == -99)
        current_variants = sum(len(cache) for cache in _slots.cache.values())
        if compiled_variants is None:
            compiled_variants = current_variants
        assert current_variants == compiled_variants
