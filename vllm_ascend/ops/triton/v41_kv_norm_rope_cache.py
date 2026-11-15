"""Experimental BF16 V4.1 SWA update. Native/graph qualification required.

This module does not patch or enable itself. It preserves the intermediate
BF16 normalization boundary, rather than rotating unrounded FP32 values.
The rope-only mode retains the existing CANN norm for a stricter comparison.
"""

import os
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




# --- KV8 精度模拟（KV8_SIM=1 开启）------------------------------------------
_KV8_SIM = os.environ.get("KV8_SIM", "0") == "1"
_KV8_SIM_GROUP = int(os.environ.get("KV8_SIM_GROUP", "64"))
_KV8_SIM_STATE = {"printed": False, "n": 0}


def _kv8_sim_quant(x):
    """per-group E4M3 量化->反量化，模拟真 KV8 的精度损失（存储仍 BF16）。"""
    if not _KV8_SIM or x is None:
        return x
    try:
        d = x.shape[-1]
        g = _KV8_SIM_GROUP
        if d % g != 0:
            return x
        xg = x.reshape(-1, d // g, g)
        amax = xg.abs().amax(-1, keepdim=True).clamp_min(1e-12)
        scale = (amax / 448.0).to(torch.bfloat16).to(x.dtype)
        q = (xg / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        out = (q.to(x.dtype) * scale).reshape(x.shape)
        if not _KV8_SIM_STATE["printed"]:
            _KV8_SIM_STATE["printed"] = True
            print(f"[KV8-SIM] quantize SWA KV shape={tuple(x.shape)} dtype={x.dtype} group={g}", flush=True)
        _KV8_SIM_STATE["n"] += 1
        return out
    except Exception as exc:
        if not _KV8_SIM_STATE["printed"]:
            _KV8_SIM_STATE["printed"] = True
            print(f"[KV8-SIM] disabled: {type(exc).__name__}: {exc}", flush=True)
        return x
# ---------------------------------------------------------------------------


def kv_norm_rope_cache(kv, norm, cos, sin, slots, cache, *, normalize=True):
    """Consume projected KV, write SWA cache in place, return no intermediate.
    kv = _kv8_sim_quant(kv)

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


# --- 可选：norm+rope 后直接量化并打包成 A3 KV8 record（W3 融合写侧）----------
# 由 `VLLM_ASCEND_A3_KV_FUSED_WRITE` 控制（auto=量化开启时生效，0=强制回退到
# mixed_quant_attention 的逐 op 打包器）。不开启时本文件既有函数逐字节不变。
#
# 三种 record（与 mixed_quant_attention.py:561-753 的 CPU 打包器逐字节同语义）：
#   fp8        544B  [0,512) E4M3 码（nope448+rope64） | [512,544) 16×BF16 scale(g32)
#   int8       608B  [0,128) rope BF16 | [128,576) INT8 码(g64) | [576,590) 7×BF16
#                    | [590,608) 18B pad
#   int8_mixed 544B  [0,448) INT8 码(g128) | [448,512) E4M3 码(g32)
#                    | [512,544) 4+2×BF16 scale + 10×1.0
# 量化语义（两条不变量，和 CPU 打包器完全一致）：
#   * scale 先用 BF16 落定，再用它量化 —— 否则读侧有纯增益误差；
#   * int8 用 RNE（ties-to-even）而非截断；fp8 走 2^k + E4M3 网格 RNE。
A3_KV_FUSED_WRITE_ENV = "VLLM_ASCEND_A3_KV_FUSED_WRITE"
_A3_FUSED_WRITE_RAW = os.environ.get(A3_KV_FUSED_WRITE_ENV, "auto").strip().lower()
# auto / 1 / true / on → 量化开启时启用融合写；0 / off / false → 强制回退
A3_FUSED_WRITE_ENABLED = _A3_FUSED_WRITE_RAW not in ("0", "off", "false", "no", "disabled")

# 与既有 bf16 融合写同一批大小上限（decode 的最大 token 数）；可用 env 放宽。
A3_FUSED_MAX_ROWS = int(os.environ.get("VLLM_ASCEND_A3_KV_FUSED_MAX_ROWS", "192"))

QFMT_FP8_544 = 1
QFMT_INT8_608 = 2
QFMT_INT8_MIXED_544 = 3
A3_RECORD_QFMT = {"fp8": QFMT_FP8_544, "int8": QFMT_INT8_608, "int8_mixed": QFMT_INT8_MIXED_544}


@triton.jit
def _rne(v):
    """round-to-nearest, ties-to-even（与 torch.round 同语义；截断有系统性偏置）。"""
    f = tl.floor(v)
    t = v - f
    odd = (f.to(tl.int32) & 1) == 1
    return f + tl.where(t > 0.5, 1.0, tl.where(t < 0.5, 0.0, tl.where(odd, 1.0, 0.0)))


@triton.jit
def _e4m3_code(v):
    """FP32 → E4M3 码（0..255，纯整数位运算，不依赖 fp8 dtype 的 TruncFOp）。

    正规数丢低 20 位尾数做 RNE；|v| < 2^-6 走 2^-9 网格（0..8，8 正好衔接最小正规）。
    与 `mixed_quant_attention._e4m3_encode` 同语义（该实现已与冻结 kernel 实测逐位一致）。
    """
    a = tl.abs(v)
    bits = a.to(tl.int32, bitcast=True)
    sticky = tl.where((bits & 0x100000) != 0, 1, 0)
    rb = (bits + 0x7FFFF + sticky) & (~0xFFFFF)
    code_n = (rb >> 20) - 960
    code_s = _rne(a * 512.0).to(tl.int32)
    code = tl.where(a >= 0.015625, code_n, code_s)  # 2^-6 = E4M3 最小正规数
    code = tl.minimum(tl.maximum(code, 0), 0x7F)
    return code + tl.where(v < 0.0, 128, 0)


@triton.jit
def _pow2_k(amax):
    """最小的整数 k 使 amax ≤ 448·2^k（448 = 2^9·0.875，纯位运算，不碰 log/除法）。"""
    bits = amax.to(tl.int32, bitcast=True)
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    k = (exp - 127 - 8) + tl.where(mant > 0x600000, 1, 0)
    k = tl.where(amax > 0.0, k, 0)
    return tl.minimum(tl.maximum(k, -120), 100)


@triton.jit
def _pow2_gain(k):
    """k → 2^k（fp32 位模式构造，逐位精确；2^-k 传 -k）。"""
    return ((127 + k) * 8388608).to(tl.float32, bitcast=True)


@triton.jit
def _group_amax(x, DIM: tl.constexpr, NG: tl.constexpr, G: tl.constexpr):
    """按 (NG,G) 切块求组内 amax：返回 [(NG,), (DIM,)] —— 小向量 + 逐 lane 广播。

    用 reshape+reduce（A3 上 reshape 是纯布局，free）而不是 gather-butterfly 或
    「masked store + d//G 偏移」：后者在 A3 上会退化成标量循环（实测慢 90x）。
    """
    m = tl.max(tl.reshape(tl.abs(x), (NG, G)), 1)
    return m, tl.reshape(tl.broadcast_to(m[:, None], (NG, G)), (DIM,))


@triton.jit(
    do_not_specialize=[
        "X_ROW", "COS_ROW", "SIN_ROW", "SLOT_ROW", "NUM_SLOTS",
        "BLOCK_ROWS", "BLOCK_STRIDE", "ROW_STRIDE",
    ]
)
def _kv_store_quant(
    X,
    GAMMA,
    BIAS,
    COS,
    SIN,
    SLOTS,
    CACHE,
    CACHE16,
    EPS: tl.constexpr,
    X_ROW,
    COS_ROW,
    SIN_ROW,
    SLOT_ROW,
    NUM_SLOTS,
    BLOCK_ROWS,
    BLOCK_STRIDE,
    ROW_STRIDE,
    DIM: tl.constexpr,
    ROPE: tl.constexpr,
    NORMALIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    QFMT: tl.constexpr,
):
    """一个 program 一个 token：norm → rope → 量化 → 打包 → 写 packed cache。

    写侧只做**仿射偏移**的 store（A3 上 masked store 的偏移一旦非仿射就会退化成
    逐 lane 标量循环）；小向量 scale 由 reshape 归约直接得到，再连续写出。
    """
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
    if ROPE > 0:
        # Pairwise adjacent rotation on only the last ROPE components.
        pair = tl.gather(y, d ^ 1, 0)
        rd = d - (DIM - ROPE)
        c = tl.load(COS + row * COS_ROW + rd, rd >= 0, other=1).to(tl.float32)
        s = tl.load(SIN + row * SIN_ROW + rd, rd >= 0, other=0).to(tl.float32)
        signed_pair = tl.where((d & 1) == 0, -pair, pair)
        rotated = y * c + signed_pair * s
        out = tl.where(rd >= 0, rotated, y)
    else:
        out = y
    # 量化器的输入是 BF16 边界之后的值（CPU 打包器看到的也是 bf16 post-RoPE）。
    xb = out.to(tl.bfloat16).to(tl.float32)
    slot = tl.load(SLOTS + row * SLOT_ROW).to(tl.int64)
    if (slot >= 0) & (slot < NUM_SLOTS):
        # flat slot = page * BLOCK_ROWS + offset；块内行连续，块间可以有洞
        # （生产池 stride(0)=131072 B 是 bf16 的 block page，544 B record 只填 69632 B）。
        page = slot // BLOCK_ROWS
        offset = slot % BLOCK_ROWS
        base = page * BLOCK_STRIDE + offset * ROW_STRIDE
        b16 = base // 2
        if QFMT == 1:  # fp8 544：nope448 + rope64 全 E4M3(g32, 2^k)
            amax, _ = _group_amax(xb, DIM, 16, 32)
            k = _pow2_k(amax)
            gain = _pow2_gain(-k)
            inv_lane = tl.reshape(tl.broadcast_to(gain[:, None], (16, 32)), (DIM,))
            code = _e4m3_code(tl.minimum(tl.maximum(xb * inv_lane, -448.0), 448.0))
            tl.store(CACHE + base + d, code.to(tl.int8))
            k16 = tl.arange(0, 16)
            tl.store(CACHE16 + b16 + 256 + k16, _pow2_gain(k).to(tl.bfloat16))
        elif QFMT == 2:  # int8 608：rope64 透传 BF16 + nope448 INT8(g64)
            xn = tl.where(d < 448, xb, 0.0)
            amax, _ = _group_amax(xn, DIM, 8, 64)
            sc = tl.maximum(amax / 127.0, 1e-12).to(tl.bfloat16).to(tl.float32)
            sl = tl.reshape(tl.broadcast_to(sc[:, None], (8, 64)), (DIM,))
            q = tl.minimum(tl.maximum(_rne(xn / sl), -127.0), 127.0).to(tl.int8)
            tl.store(CACHE + base + 128 + d, q, mask=d < 448)
            # rope 段 BF16 透传：先在寄存器里 gather 成 64 lane 再连续写。
            # A3 上 masked store 的 fp32→bf16 转换是 1 ULP 偏差（实测 140/512 lane
            # 与寄存器 cast 不同），所以不在 store 处降精度。
            r64 = tl.arange(0, 64)
            tl.store(CACHE16 + b16 + r64, tl.gather(xb, 448 + r64, 0).to(tl.bfloat16))
            k8 = tl.arange(0, 8)
            sc8 = tl.where(k8 < 7, sc, 0.0)   # 第 8 槽落在 pad 区，写 0
            tl.store(CACHE16 + b16 + 288 + k8, sc8.to(tl.bfloat16))
            tl.store(CACHE + base + 592 + tl.arange(0, 16), tl.zeros((16,), tl.int8))
        else:  # int8_mixed 544：nope448 INT8(g128) + rope64 E4M3(g32)
            xn = tl.where(d < 448, xb, 0.0)
            amax_n, _ = _group_amax(xn, DIM, 4, 128)
            sc_n = tl.maximum(amax_n / 127.0, 1e-12).to(tl.bfloat16).to(tl.float32)
            sl = tl.reshape(tl.broadcast_to(sc_n[:, None], (4, 128)), (DIM,))
            qn = tl.minimum(tl.maximum(_rne(xn / sl), -127.0), 127.0).to(tl.int8)
            xr = tl.where(d >= 448, xb, 0.0)
            amax_r, _ = _group_amax(xr, DIM, 16, 32)   # 只有第 14/15 组是 rope
            k = _pow2_k(amax_r)
            gain_lane = tl.reshape(tl.broadcast_to(_pow2_gain(k)[:, None], (16, 32)), (DIM,))
            inv_lane = tl.reshape(tl.broadcast_to(_pow2_gain(-k)[:, None], (16, 32)), (DIM,))
            cr = _e4m3_code(tl.minimum(tl.maximum(xr * inv_lane, -448.0), 448.0)).to(tl.int8)
            # 码段必须**一次**写完（两个互补 mask 的 store 打到同一 512B 区间时，
            # A3 后端会重排 → 实测非确定性地互相覆盖）。
            tl.store(CACHE + base + d, tl.where(d < 448, qn, cr))
            # scale 区 32B 对齐、一次写完：[n0..n3, r0, r1, 1.0×10]
            # 用 gather 从逐 lane scale 取 6 个值（实测比 (16,16) tile 求和快 2.4x）
            idx = tl.arange(0, 16)
            sl_lane = tl.where(d < 448, sl, gain_lane)
            src = tl.where(idx < 4, idx * 128, tl.where(idx < 6, 448 + (idx - 4) * 32, 0))
            v16 = tl.where(idx < 6, tl.gather(sl_lane, src, 0), 1.0)
            tl.store(CACHE16 + b16 + 256 + idx, v16.to(tl.bfloat16))


def _packed_pool_layout(cache):
    """packed record 池 → (num_slots, block_rows, block_stride, row_stride)，否则 None。

    A3 KV8 的生产池是 ``(blocks, block_rows, 1, slot_bytes)`` 且 ``stride(0)`` 是
    **沿用 bf16 的 block page**（131072 B），而 544 B record 一页只填
    ``block_rows * 544`` B（128 × 544 = 69632）⇒ 块间有 61440 B 的洞，
    ``cache.is_contiguous() == False``。但「块内行连续、行内字节连续」时，
    按 ``page * stride(0) + offset * stride(1)`` 就能正确寻址（W7 GRAPH_SLOT.md §1.5），
    所以判据不该是 is_contiguous()，而是这个 stride 结构。
    """
    if cache.dim() < 2 or int(cache.stride(-1)) != 1:
        return None
    slot_bytes = int(cache.shape[-1])
    if slot_bytes not in (544, 608):
        return None
    if cache.is_contiguous():
        rows = int(cache.numel()) // slot_bytes
        if rows <= 0:
            return None
        # 连续池：等价于「一个 block 装下全部行」，block_stride 不参与寻址。
        return rows, rows, 0, slot_bytes
    if cache.dim() == 4 and int(cache.shape[2]) == 1:
        block_rows = int(cache.shape[1])
        row_stride = int(cache.stride(1))
        block_stride = int(cache.stride(0))
        if block_rows <= 0 or row_stride != slot_bytes:
            return None
        if block_stride < block_rows * row_stride:
            return None
        return block_rows * int(cache.shape[0]), block_rows, block_stride, row_stride
    return None


def supports_quant(kv, cos, sin, flat_slots, cache, *, normalize=True, rope=64):
    """Metadata-only dispatch for the fused quantized write (no device reads, no sync)."""
    if not A3_FUSED_WRITE_ENABLED or flat_slots is None:
        return False
    if cache.dim() < 2 or int(cache.stride(-1)) != 1:
        return False
    # 判据从 is_contiguous() 改成「按真实步长可寻址」：生产池块间带洞
    # （stride(0)=131072 > block_rows*544），但 page/offset 寻址是正确的。
    if _packed_pool_layout(cache) is None:
        return False
    if int(cache.shape[-1]) not in (544, 608):
        return False
    if cache.dtype not in (torch.int8, torch.uint8):
        return False
    if rope > 0 and cos is not None:
        if cos.shape != sin.shape or cos.shape != (kv.shape[0], 1, 1, 64):
            return False
        if cos.stride(-1) != 1 or sin.stride(-1) != 1 or cos.dtype != sin.dtype:
            return False
    return (
        kv.ndim == 2
        and kv.shape[1] == 512
        and kv.dtype == torch.bfloat16
        and kv.stride(-1) == 1
        and 0 < kv.shape[0] <= A3_FUSED_MAX_ROWS
        and flat_slots.ndim == 1
        and flat_slots.numel() >= kv.shape[0]
        and flat_slots.stride(-1) == 1
        and flat_slots.dtype in (torch.int32, torch.int64)
        and kv.device == flat_slots.device == cache.device
    )


def kv_norm_rope_cache_quant(kv, norm, cos, sin, flat_slots, cache, *, qfmt, normalize=True, rope=64):
    """norm(可选) + rope(可选) + 量化 + 打包 + 写 packed cache，一次 kernel 完成。

    ``qfmt`` 取 ``"fp8"`` / ``"int8"`` / ``"int8_mixed"``（或 QFMT_* 常量）。
    ``rope=0`` 表示输入已经是 post-RoPE 的值（跳过旋转，其余不变）。
    量化语义与 ``mixed_quant_attention`` 的 CPU 打包器逐字节一致。
    """
    if isinstance(qfmt, str):
        qfmt = A3_RECORD_QFMT[qfmt]
    if not supports_quant(kv, cos, sin, flat_slots, cache, normalize=normalize, rope=rope):
        raise ValueError("Unsupported fused quantized KV write metadata; use original branch")
    slot_bytes = int(cache.shape[-1])
    if qfmt == QFMT_INT8_608 and slot_bytes != 608:
        raise ValueError(f"int8 608 record needs a 608 B slot, got {slot_bytes}")
    if qfmt in (QFMT_FP8_544, QFMT_INT8_MIXED_544) and slot_bytes != 544:
        raise ValueError(f"{qfmt} record needs a 544 B slot, got {slot_bytes}")
    layout = _packed_pool_layout(cache)
    if layout is None:
        raise ValueError("Unsupported fused quantized KV write metadata; use original branch")
    num_slots, block_rows, block_stride, row_stride = layout
    weight = norm.weight if norm is not None else cache
    bias = getattr(norm, "bias", None)
    has_bias = bias is not None and getattr(norm, "bias_loaded", False)
    t = kv.shape[0]
    _kv_store_quant[(t,)](
        kv,
        weight,
        bias if has_bias else weight,
        cos if rope > 0 else weight,
        sin if rope > 0 else weight,
        flat_slots,
        # 注意：**不能** .reshape(-1) —— 块间带洞时 reshape 会拷贝，写进临时张量就丢了。
        # Triton 只用 data_ptr（忽略 strides），偏移由 kernel 按 page/offset 自己算。
        cache.view(torch.int8),
        cache.view(torch.bfloat16),
        float(getattr(norm, "variance_epsilon", 1e-6)),
        kv.stride(0),
        cos.stride(0) if rope > 0 else 1,
        sin.stride(0) if rope > 0 else 1,
        flat_slots.stride(0),
        num_slots,
        block_rows,
        block_stride,
        row_stride,
        512,
        rope,
        normalize,
        has_bias,
        qfmt,
        enable_fp_fusion=False,
    )
