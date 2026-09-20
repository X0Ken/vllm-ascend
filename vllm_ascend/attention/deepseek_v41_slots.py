# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused slot coordinates; no changes to cache ownership or update policy."""

from vllm.triton_utils import tl, triton


# Batch lengths are runtime metadata, not separate kernel variants.
@triton.jit(do_not_specialize=["N", "ACTUAL_REQS", "ACTUAL_TOKENS"])
def _slots(
    raw,
    pos,
    qsl,
    output,
    N,
    ACTUAL_REQS,
    ACTUAL_TOKENS,
    RATIO: tl.constexpr,
    COMPRESSED: tl.constexpr,
    HAS_POS: tl.constexpr,
    SKIP: tl.constexpr,
    STORAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(raw + i, i < N, other=-1).to(tl.int64)
    valid = x >= 0
    if COMPRESSED and RATIO == 2:
        valid = valid & ((x + 1) % 2 == 0)
        x = x // 2
        end = tl.minimum(tl.load(qsl + ACTUAL_REQS), ACTUAL_TOKENS)
        valid = valid & (i < end)
        if HAS_POS:
            p = tl.load(pos + i, i < N, other=0)
            valid = valid & (p % 2 == 1)
        if SKIP:
            valid = tl.full((BLOCK,), False, tl.int1)
    safe = tl.maximum(x, 0)
    row = tl.where(valid, safe // STORAGE, -1)
    col = tl.where(valid, safe % STORAGE, -1)
    tl.store(output + 2 * i, row, i < N)
    tl.store(output + 2 * i + 1, col, i < N)


def slot_coordinates(builder, common, positions, n, actual_reqs, actual_tokens, ratio, compressed, skip):
    out = builder._slot_mapping_2d[:n]
    if n:
        _slots[(triton.cdiv(n, 128),)](
            common.slot_mapping,
            positions if positions is not None else common.slot_mapping,
            common.query_start_loc,
            out,
            n,
            actual_reqs,
            actual_tokens,
            ratio,
            compressed,
            positions is not None,
            skip,
            builder.kv_cache_spec.storage_block_size,
            128,
        )
    return out
