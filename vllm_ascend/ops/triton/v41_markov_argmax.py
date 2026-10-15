"""Greedy replicated Markov correction with the original BF16 add boundary.

No communication, Markov matmul, or serial draft-step dependency changes.
The caller must have no consumer of corrected logits (greedy DSpark only).
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _partials(
    X,
    Y,
    VALUES,
    IDS,
    NANS,
    V: tl.constexpr,
    XS: tl.constexpr,
    YS: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    col = part * BLOCK + tl.arange(0, BLOCK)
    valid = col < V
    x = tl.load(X + row * XS + col, valid, other=0).to(tl.float32)
    y = tl.load(Y + row * YS + col, valid, other=0).to(tl.float32)
    # In-place torch BF16 add stores rounded values before argmax reads them.
    z = (x + y).to(tl.bfloat16).to(tl.float32)
    isnan = (z != z) & valid
    finite_or_inf = tl.where(valid & ~isnan, z, -float("inf"))
    maximum = tl.max(finite_or_inf, 0)
    best = tl.min(tl.where(valid & ~isnan & (z == maximum), col, 2147483647), 0)
    first_nan = tl.min(tl.where(isnan, col, 2147483647), 0)
    offset = row * PARTS + part
    tl.store(VALUES + offset, maximum)
    tl.store(IDS + offset, best)
    tl.store(NANS + offset, first_nan)


@triton.jit
def _finish(VALUES, IDS, NANS, OUT, PARTS: tl.constexpr, OS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    valid = part < PARTS
    value = tl.load(VALUES + row * PARTS + part, valid, other=-float("inf"))
    index = tl.load(IDS + row * PARTS + part, valid, other=2147483647)
    nan_index = tl.load(NANS + row * PARTS + part, valid, other=2147483647)
    maximum = tl.max(value, 0)
    best = tl.min(tl.where(valid & (value == maximum), index, 2147483647), 0)
    first_nan = tl.min(nan_index, 0)
    # Native gate checks this against actual NPU ArgMaxV2 including NaNs.
    result = tl.where(first_nan != 2147483647, first_nan, best)
    tl.store(OUT + row * OS, result.to(tl.int64))


def supported(logits, bias, output):
    return (
        logits.ndim == bias.ndim == 2
        and logits.shape == bias.shape
        and logits.dtype == bias.dtype == torch.bfloat16
        and logits.device.type == "npu"
        and logits.device == bias.device == output.device
        and logits.stride(1) == bias.stride(1) == 1
        and output.ndim == 1
        and output.shape[0] == logits.shape[0]
        and output.stride(0) > 0
        and output.dtype == torch.int64
        and 0 < logits.shape[0] <= 32
        and logits.shape[1] == 129280
    )


def add_argmax_out(logits, bias, output):
    assert supported(logits, bias, output)
    rows, vocab = logits.shape
    block = 1024
    parts = triton.cdiv(vocab, block)
    values = torch.empty((rows, parts), device=logits.device, dtype=torch.float32)
    ids = torch.empty((rows, parts), device=logits.device, dtype=torch.int32)
    nans = torch.empty_like(ids)
    _partials[(rows, parts)](
        logits, bias, values, ids, nans, vocab, logits.stride(0), bias.stride(0), parts, block, enable_fp_fusion=False
    )
    _finish[(rows,)](
        values, ids, nans, output, parts, output.stride(0), triton.next_power_of_2(parts), enable_fp_fusion=False
    )
    return output
