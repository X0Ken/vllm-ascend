# SPDX-License-Identifier: Apache-2.0
"""Independent tensor reference for slot coordinate tests on CPU and NPU."""

import torch


def reference_slot_coordinates(builder, common, positions, n, actual_reqs, actual_tokens, ratio, compressed, skip):
    slots = common.slot_mapping[:n].clone()
    valid = slots >= 0
    if compressed and ratio != 1:
        valid &= (slots + 1) % ratio == 0
        slots = slots // ratio
    if compressed and ratio == 2:
        end = min(int(common.query_start_loc[actual_reqs]), actual_tokens)
        valid &= torch.arange(n, device=slots.device) < end
        if positions is not None:
            valid &= positions[:n] % 2 == 1
        if skip:
            valid.zero_()
    storage = builder.kv_cache_spec.storage_block_size
    builder._slot_mapping_2d[:n, 0] = torch.where(valid, slots.clamp_min(0) // storage, -1)
    builder._slot_mapping_2d[:n, 1] = torch.where(valid, slots.clamp_min(0) % storage, -1)
    if getattr(builder, "_flat_slot_mapping", None) is not None:
        builder._flat_slot_mapping[:n] = torch.where(valid, slots, -1)
    return builder._slot_mapping_2d[:n]
