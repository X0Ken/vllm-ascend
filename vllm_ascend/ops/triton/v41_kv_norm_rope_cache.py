"""Experimental BF16 V4.1 SWA update. Native/graph qualification required.

This module does not patch or enable itself. It preserves the intermediate
BF16 normalization boundary, rather than rotating unrounded FP32 values.
The rope-only mode retains the existing CANN norm for a stricter comparison.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["X_ROW", "COS_ROW", "SIN_ROW", "SLOT_ROW", "PAGE_STRIDE", "TOKEN_STRIDE"])
def _kv_store(
    X,
    GAMMA,
    BIAS,
    COS,
    SIN,
    SLOTS,
    CACHE,
    EPS: tl.constexpr,
    X_ROW,
    COS_ROW,
    SIN_ROW,
    SLOT_ROW,
    PAGE_STRIDE,
    TOKEN_STRIDE,
    PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIM: tl.constexpr,
    ROPE: tl.constexpr,
    NORMALIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, DIM)
    x = tl.load(X + row * X_ROW + d).to(tl.float32)
    if NORMALIZE:
        var = tl.sum(x * x, 0) / DIM
        inv = tl.rsqrt(var + EPS)
        gamma = tl.load(GAMMA + d).to(tl.float32)
        y = ((x * inv) * gamma).to(X.dtype.element_ty).to(tl.float32)
        if HAS_BIAS:
            bias = tl.load(BIAS + d).to(tl.float32)
            y = (y + bias).to(X.dtype.element_ty).to(tl.float32)
    else:
        y = x
    # Pairwise adjacent rotation on only the last ROPE components.
    pair = tl.gather(y, d ^ 1, 0)
    rd = d - (DIM - ROPE)
    c = tl.load(COS + row * COS_ROW + rd, rd >= 0, other=1).to(tl.float32)
    s = tl.load(SIN + row * SIN_ROW + rd, rd >= 0, other=0).to(tl.float32)
    signed_pair = tl.where((d & 1) == 0, -pair, pair)
    rotated = y * c + signed_pair * s
    out = tl.where(rd >= 0, rotated, y).to(CACHE.dtype.element_ty)
    page = tl.load(SLOTS + row * SLOT_ROW).to(tl.int64)
    offset = tl.load(SLOTS + row * SLOT_ROW + 1).to(tl.int64)
    valid = (page >= 0) & (page < PAGES) & (offset >= 0) & (offset < PAGE_SIZE)
    dest = CACHE + page * PAGE_STRIDE + offset * TOKEN_STRIDE + d
    tl.store(dest, out, valid)


def supports(kv, norm, cos, sin, slots, cache):
    """Metadata-only dispatch; no device reads, scalar extraction or sync."""
    plane = cache.squeeze(-2)
    return (
        kv.ndim == 2
        and kv.shape[1] == 512
        and kv.dtype == torch.bfloat16
        and kv.stride(-1) == 1
        and 0 < kv.shape[0] <= 192
        and norm.weight.shape == (512,)
        and norm.weight.is_contiguous()
        and cos.shape == sin.shape == (kv.shape[0], 1, 1, 64)
        and cos.stride(-1) == sin.stride(-1) == 1
        and cos.dtype == sin.dtype
        and cos.dtype in (torch.bfloat16, torch.float32)
        and slots.ndim == 2
        and slots.shape[1] == 2
        and slots.shape[0] >= kv.shape[0]
        and slots.stride(-1) == 1
        and slots.dtype in (torch.int32, torch.int64)
        and plane.ndim == 3
        and plane.shape[-1] == 512
        and plane.stride(-1) == 1
        and plane.dtype == torch.bfloat16
        and kv.device == cos.device == sin.device == slots.device == cache.device
    )


def kv_norm_rope_cache(kv, norm, cos, sin, slots, cache, *, normalize=True):
    """Consume projected KV, write SWA cache in place, return no intermediate.

    Slots must be unique among valid rows (the existing scatter's contract).
    Negative dummy slots are skipped; dynamic device slots are graph inputs.
    For normalize=False the caller passes CANN-normalized BF16 KV.
    """
    if not supports(kv, norm, cos, sin, slots, cache):
        raise ValueError("Unsupported KV fusion metadata; use original branch")
    plane = cache.squeeze(-2)
    bias = getattr(norm, "bias", None)
    has_bias = bias is not None and getattr(norm, "bias_loaded", False)
    _kv_store[(kv.shape[0],)](
        kv,
        norm.weight,
        bias if has_bias else norm.weight,
        cos,
        sin,
        slots,
        plane,
        norm.variance_epsilon,
        kv.stride(0),
        cos.stride(0),
        sin.stride(0),
        slots.stride(0),
        plane.stride(0),
        plane.stride(1),
        plane.shape[0],
        plane.shape[1],
        512,
        64,
        normalize,
        has_bias,
        enable_fp_fusion=False,
    )
