# SPDX-License-Identifier: Apache-2.0
"""Compare fused slot coordinates with an independent tensor reference on A2."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from tests.deepseek_v41_slots import reference_slot_coordinates
from vllm_ascend.attention.deepseek_v41_slots import slot_coordinates


@pytest.mark.parametrize("n", [0, 1, 7, 129, 8192])
@pytest.mark.parametrize("ratio", [1, 2, 128])
@pytest.mark.parametrize("compressed,skip", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("with_positions", [False, True])
def test_slot_coordinates(n, ratio, compressed, skip, with_positions):
    size = n + 3
    raw = torch.arange(size, dtype=torch.int64) * 63 - 1
    raw[::7] = -1
    positions = torch.arange(size, dtype=torch.int64) if with_positions else None
    cpu_common = SimpleNamespace(slot_mapping=raw, query_start_loc=torch.tensor([0, max(n - 2, 0)]))
    common = SimpleNamespace(slot_mapping=raw.npu(), query_start_loc=cpu_common.query_start_loc.npu())

    def builder(device):
        return SimpleNamespace(
            _slot_mapping_2d=torch.full((size, 2), -99, dtype=torch.int32, device=device),
            _flat_slot_mapping=torch.full((size,), -99, dtype=torch.int64, device=device),
            kv_cache_spec=SimpleNamespace(storage_block_size=64),
        )

    reference, actual = builder("cpu"), builder("npu")
    reference_slot_coordinates(reference, cpu_common, positions, n, 1, n, ratio, compressed, skip)
    slot_coordinates(
        actual, common, positions.npu() if positions is not None else None, n, 1, n, ratio, compressed, skip
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual._slot_mapping_2d.cpu(), reference._slot_mapping_2d, rtol=0, atol=0)
    torch.testing.assert_close(actual._flat_slot_mapping.cpu(), reference._flat_slot_mapping, rtol=0, atol=0)
