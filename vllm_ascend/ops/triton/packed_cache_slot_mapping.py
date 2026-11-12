# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build packed cache slot metadata in one NPU launch."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


def _is_pow2(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "num_actual_reqs",
        "num_actual_tokens",
        "skip_update",
    ]
)
def _build_packed_cache_slot_mapping_kernel(
    slots_ptr,
    positions_ptr,
    query_start_loc_ptr,
    coordinates_ptr,
    flat_slots_ptr,
    num_tokens,
    num_actual_reqs,
    num_actual_tokens,
    skip_update,
    slot_stride,
    position_stride,
    query_start_stride,
    PAGE_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    POW2_PAGE: tl.constexpr,
    PAGE_LOG2: tl.constexpr,
    NARROW_SLOTS: tl.constexpr,
):
    tokens = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = tokens < num_tokens
    slots = tl.load(slots_ptr + tokens * slot_stride, mask=mask, other=-1)
    # Physical slot ids are bounded by the number of KV slots in this engine
    # (a few million at 1M context), so 32-bit arithmetic is exact here. This
    # kernel is pure AIV scalar work -- 89% of its device time is the scalar
    # pipe -- and A2's scalar unit emulates 64-bit integer division, which
    # dominated a fused launch that only moves a few hundred bytes. Narrowing
    # the operands is what makes this a single cheap launch instead of a
    # ~50 us one. When the caller cannot offer a 32-bit tensor the kernel
    # keeps 64-bit arithmetic; only the shift/div choice below still applies.
    if NARROW_SLOTS:
        slots = slots.to(tl.int32)
        request_end = tl.load(query_start_loc_ptr + num_actual_reqs * query_start_stride).to(tl.int32)
        num_actual_tokens32 = num_actual_tokens.to(tl.int32)
        valid_end = tl.minimum(request_end, num_actual_tokens32)
    else:
        slots = slots.to(tl.int64)
        request_end = tl.load(query_start_loc_ptr + num_actual_reqs * query_start_stride).to(tl.int64)
        valid_end = tl.minimum(request_end, num_actual_tokens)
    # Full graphs keep a fixed token carrier whose padded tail may still hold
    # slot IDs from an earlier replay.  Never let those rows write any packed cache
    # plane, including the uncompressed SWA/C1 planes.
    valid = mask & (tokens < valid_end) & (slots >= 0)

    if COMPRESS_RATIO == 2:
        # ``& 1`` is ``% 2`` and ``>> 1`` is floor-division by two, including
        # for the negative sentinel rows that the ``slots >= 0`` filter above
        # has already excluded from every store.
        valid &= (slots & 1) == 1
        positions = tl.load(
            positions_ptr + tokens * position_stride,
            mask=mask,
            other=0,
        ).to(tl.int64)
        valid &= ((positions % 2) == 1) & (skip_update == 0)
        physical = slots >> 1
    else:
        physical = slots

    safe_physical = tl.maximum(physical, 0)
    if POW2_PAGE:
        pages = tl.where(valid, safe_physical >> PAGE_LOG2, -1)
        rows = tl.where(valid, safe_physical & (PAGE_SIZE - 1), -1)
    else:
        pages = tl.where(valid, safe_physical // PAGE_SIZE, -1)
        rows = tl.where(valid, safe_physical % PAGE_SIZE, -1)
    flat_slots = tl.where(valid, physical, -1)
    tl.store(flat_slots_ptr + tokens, flat_slots, mask=mask)
    tl.store(coordinates_ptr + tokens * 2, pages, mask=mask)
    tl.store(coordinates_ptr + tokens * 2 + 1, rows, mask=mask)


def build_packed_cache_slot_mapping(
    slots: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_tokens: int,
    num_actual_reqs: int,
    num_actual_tokens: int,
    page_size: int,
    compress_ratio: int,
    *,
    skip_update: bool = False,
    coordinates_output: torch.Tensor,
    flat_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Publish coordinate and flat packed cache addresses together.

    This replaces the eager GE/Add/Remainder/And/Div/Where/Copy chain used by
    every KV cache group. Ratio-2 completion and padded-row masking follow the
    existing metadata builder contract exactly.
    """
    if compress_ratio not in (1, 2):
        raise ValueError("compress_ratio must be 1 or 2")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if num_tokens:
        block_size = 256
        pow2_page = _is_pow2(page_size)
        _build_packed_cache_slot_mapping_kernel[(triton.cdiv(num_tokens, block_size),)](
            slots,
            positions,
            query_start_loc,
            coordinates_output,
            flat_output,
            num_tokens,
            num_actual_reqs,
            num_actual_tokens,
            int(skip_update),
            slots.stride(0),
            positions.stride(0),
            query_start_loc.stride(0),
            PAGE_SIZE=page_size,
            COMPRESS_RATIO=compress_ratio,
            BLOCK_SIZE=block_size,
            POW2_PAGE=pow2_page,
            PAGE_LOG2=page_size.bit_length() - 1 if pow2_page else 0,
            NARROW_SLOTS=slots.dtype == torch.int32,
        )
    return coordinates_output, flat_output
