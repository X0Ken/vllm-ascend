# SPDX-License-Identifier: Apache-2.0
"""Fuse expert-map lookup, ownership predicate, and router-weight masking."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _mask_weights(
    W,
    IDS,
    MAP,
    OUT,
    N: tl.constexpr,
    K: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    IS0: tl.constexpr,
    IS1: tl.constexpr,
    MS: tl.constexpr,
    EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offset < N
    row, column = offset // K, offset % K
    expert = tl.load(IDS + row * IS0 + column * IS1, active, other=0).to(tl.int32)
    # Preserve Python tensor-index semantics for valid negative indices too.
    expert = tl.where(expert < 0, expert + EXPERTS, expert)
    owner = tl.load(MAP + expert * MS, active, other=-1)
    weight = tl.load(W + row * WS0 + column * WS1, active, other=0).to(tl.float32)
    value = weight * (owner != -1).to(tl.float32)
    tl.store(OUT + offset, value, active)


def mask_expert_weights(weights, expert_ids, expert_map):
    """Keep the exact original multiplication; preserve quantization/GMM ABI."""
    assert weights.ndim == expert_ids.ndim == 2 and weights.shape == expert_ids.shape
    if weights.dtype not in (torch.float32, torch.bfloat16):
        return weights * (expert_map[expert_ids.long()] != -1)
    output = torch.empty(weights.shape, device=weights.device, dtype=weights.dtype)
    count = weights.numel()
    if count:
        _mask_weights[(triton.cdiv(count, 256),)](
            weights,
            expert_ids,
            expert_map,
            output,
            count,
            weights.shape[1],
            *weights.stride(),
            *expert_ids.stride(),
            expert_map.stride(0),
            expert_map.numel(),
            BLOCK=256,
            num_warps=4,
        )
    return output
