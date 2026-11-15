# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed-quant cache and attention ABI selected by the A5 device adaptor."""

from __future__ import annotations

import os as _os
import sys as _sys

import torch

from vllm_ascend.models.deepseek_v41.cache_config import (
    A3_KV_DTYPE_FP8,
    A3_KV_DTYPE_INT8,
    A3_KV_DTYPE_INT8_MIXED,
    a3_kv_dtype,
    kv_slot_format,
    kv_slot_quant_mode,
)
from vllm_ascend.ops import packaged_attention as ops
from vllm_ascend.ops.triton.build_window_indices import (
    build_window_indices_triton,
)
from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.packed_cache_slot_mapping import build_packed_cache_slot_mapping
from vllm_ascend.ops.triton.prepare_indexer_indices import (
    prepare_indexer_indices,
)
from vllm_ascend.ops.triton.quantize_mxfp4_indexer import (
    quantize_mxfp4_indexer,
    write_mxfp4_indexer_cache,
)
from vllm_ascend.worker.device_metadata import DeviceMetadataStage, wait_for_device_metadata


def _common(query, weights, source_cache, source_metadata, compress_ratio, quantized_query, query_scale):
    op_metadata = source_metadata.qli_metadata
    if op_metadata is None:
        raise RuntimeError("A5 QLI metadata was not built")
    wait_for_device_metadata(DeviceMetadataStage.INDEXER, id(op_metadata))
    if quantized_query is None:
        quantized_query, query_scale = quantize_mxfp4_indexer(query)
    query_start_loc = source_metadata.query_start_loc
    return (
        quantized_query,
        query_scale.unflatten(-1, (2, 2)).contiguous(),
        weights.float().contiguous(),
        source_cache[0],
        source_cache[1].unflatten(-1, (2, 2)),
        dict(
            cu_seqlens_q=query_start_loc,
            seqused_k=source_metadata.cache_seq_lens,
            cmp_residual_k=(source_metadata.cmp_residual if compress_ratio != 1 else None),
            block_table=source_metadata.block_table,
            metadata=op_metadata,
            max_seqlen_q=-1,
            mask_mode=3,
            cmp_ratio=compress_ratio,
            layout_q="TND",
            return_value=False,
        ),
    )


def _qli(
    query,
    weights,
    positions,
    source_cache,
    source_metadata,
    *,
    topk,
    compress_ratio,
    is_candidate_source,
    candidate_topk_blocks,
    candidate_block_size,
    candidates,
    candidate_lengths,
    topk_lengths,
    indices_output,
    quantized_query,
    query_scale,
):
    q, qs, w, k, ks, common = _common(
        query,
        weights,
        source_cache,
        source_metadata,
        compress_ratio,
        quantized_query,
        query_scale,
    )
    common["layout_k"] = "PA_BBND"
    indices, _, candidate_out, candidate_length = ops.quant_lightning_indexer(
        q,
        k,
        w,
        qs,
        ks,
        topk,
        1,
        candidate_topk_blocks=(candidate_topk_blocks if is_candidate_source else -1),
        candidate_block_size=(candidate_block_size if is_candidate_source else -1),
        **common,
    )
    if is_candidate_source:
        candidate_lengths.copy_(candidate_length)
    selected = prepare_indexer_indices(
        indices.squeeze(1),
        positions,
        compress_ratio,
        lengths_output=topk_lengths,
        indices_output=indices_output,
    )
    return selected, candidate_out if is_candidate_source else candidates


def _qsli(
    query,
    weights,
    positions,
    source_cache,
    source_metadata,
    *,
    topk,
    compress_ratio,
    candidate_block_size,
    candidates,
    candidate_lengths,
    topk_lengths,
    indices_output,
    quantized_query,
    query_scale,
):
    if len(source_cache) != 3:
        raise RuntimeError("A5 QSLI requires its source's folded K/scale twin")
    q, qs, w, _, _, common = _common(
        query,
        weights,
        source_cache,
        source_metadata,
        compress_ratio,
        quantized_query,
        query_scale,
    )
    common.pop("metadata")
    metadata = ops.quant_sparse_lightning_indexer_metadata(
        candidate_lengths,
        cu_seqlens_q=common["cu_seqlens_q"],
        seqused_k=common["seqused_k"],
        cmp_residual_k=common["cmp_residual_k"],
        batch_size=source_metadata.num_reqs,
        max_seqlen_q=-1,
        max_seqlen_k=-1,
        num_heads_q=q.shape[1],
        num_heads_k=1,
        head_dim=q.shape[2] * 2,
        topk=topk,
        quant_mode=1,
        candidate_block_size=candidate_block_size,
        mask_mode=3,
        cmp_ratio=compress_ratio,
        layout_q="TND",
        layout_k="PA_BBND",
    )
    common["metadata"] = metadata
    common["layout_k"] = "PA_BBND"
    indices, _ = ops.quant_sparse_lightning_indexer(
        q,
        source_cache[2].squeeze(2),
        w,
        qs,
        candidates,
        candidate_lengths,
        topk,
        1,
        candidate_block_size,
        **common,
    )
    selected = prepare_indexer_indices(
        indices.squeeze(1),
        positions,
        compress_ratio,
        lengths_output=topk_lengths,
        indices_output=indices_output,
    )
    return selected, candidates


def run_a5_indexer(
    query,
    weights,
    positions,
    source_cache,
    source_metadata,
    *,
    topk,
    compress_ratio,
    is_candidate_source,
    uses_candidate_filter,
    candidate_topk_blocks,
    candidate_block_size,
    candidates,
    candidate_lengths,
    topk_lengths,
    indices_output,
    quantized_query=None,
    query_scale=None,
):
    if uses_candidate_filter:
        return _qsli(
            query,
            weights,
            positions,
            source_cache,
            source_metadata,
            topk=topk,
            compress_ratio=compress_ratio,
            candidate_block_size=candidate_block_size,
            candidates=candidates,
            candidate_lengths=candidate_lengths,
            topk_lengths=topk_lengths,
            indices_output=indices_output,
            quantized_query=quantized_query,
            query_scale=query_scale,
        )
    return _qli(
        query,
        weights,
        positions,
        source_cache,
        source_metadata,
        topk=topk,
        compress_ratio=compress_ratio,
        is_candidate_source=is_candidate_source,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        candidates=candidates,
        candidate_lengths=candidate_lengths,
        topk_lengths=topk_lengths,
        indices_output=indices_output,
        quantized_query=quantized_query,
        query_scale=query_scale,
    )


def _qli_metadata_a3(
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    batch_size=None,
    max_seqlen_q=-1,
    max_seqlen_k=-1,
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    topk: int,
    quant_mode: int = 1,
    mask_mode: int = 0,
    cmp_ratio: int = 1,
    layout_q: str = "TND",
    layout_k: str = "TND",
    candidate_topk_blocks: int = -1,
    candidate_block_size: int = -1,
):
    """把 vllm_ascend:: 的 QLI metadata 参数适配到 cann_ops_transformer 的 schema
    （多了 quant_mode，且 num_heads_q/k/head_dim/topk 在前）。"""
    return torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata(
        num_heads_q,
        num_heads_k,
        head_dim,
        topk,
        quant_mode,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        layout_q=layout_q,
        layout_k=layout_k,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
    )


def _smla_metadata_a3(
    ori_topk_length,
    cmp_topk_length,
    *,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_ori_kv=None,
    max_seqlen_cmp_kv=None,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    quant_mode: int,
    layout_q: str = "TND",
    layout_kv: str = "PA_BBND",
    has_ori_kv: bool = True,
    has_cmp_kv: bool = True,
    # ↓ A3 起必须与 attention 侧同源：mask/窗口/ratio/topk 都会改 tiling。
    #   下列 kwargs 在算子 schema 里全都存在（容器内实测 _schema，
    #   见 TI/VLLM_PATCH.md §4）。
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 0,
    cmp_ratio: int = 1,
    ori_win_left: int = 0,
    ori_win_right: int = 0,
    ori_topk: int = 0,
    cmp_topk: int = 0,
    rope_head_dim: int = 64,
):
    """把 vllm_ascend:: 那套（topk_length 位置在后）适配到 cann_ops_transformer 的 schema
    （num_heads_q/kv/head_dim/quant_mode 在前位置，topk_length 走关键字）。

    mask/win/ratio/topk 必须与同一次 attention 调用一致：metadata 的 AICPU 核按
    ``quant_mode`` 选 ``s2BaseSize_ = quant_mode==3 ? 512 : 128``
    （mixed_quant_sparse_flash_mla_metadata/op_kernel_aicpu/..._aicpu.cpp:444），
    错配会直接错 tiling。"""
    return torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla_metadata(
        num_heads_q,
        num_heads_kv,
        head_dim,
        quant_mode,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_ori_kv=max_seqlen_ori_kv,
        max_seqlen_cmp_kv=max_seqlen_cmp_kv,
        ori_topk_length=ori_topk_length,
        cmp_topk_length=cmp_topk_length,
        ori_topk=ori_topk,
        cmp_topk=cmp_topk,
        rope_head_dim=rope_head_dim,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        has_ori_kv=has_ori_kv,
        has_cmp_kv=has_cmp_kv,
    )


def build_smla_metadata(length_rows: torch.Tensor, cu_seqlens_q: torch.Tensor) -> torch.Tensor:
    """Build the fixed A5 mixed-quant SMLA launch metadata."""
    return _smla_metadata_a3(
        length_rows,
        length_rows,
        cu_seqlens_q=cu_seqlens_q,
        num_heads_q=64,
        num_heads_kv=1,
        head_dim=512,
        quant_mode=3,
        layout_q="TND",
        layout_kv="PA_BBND",
        has_ori_kv=True,
        has_cmp_kv=True,
        cmp_ratio=4,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=63,
        ori_win_right=0,
    )


def _resolve_window_indices(q, metadata, window_size):
    indices = metadata.swa.ori_sparse_indices
    lengths = metadata.swa.ori_topk_length
    if indices is None or lengths is None:
        return build_window_indices_triton(
            metadata.positions[: q.shape[0]],
            window_size,
        )
    return indices[: q.shape[0]], lengths[: q.shape[0]]


@torch._dynamo.disable
def qsmla(
    q,
    ori_kv,
    cmp_kv,
    metadata,
    compressed_indices,
    *,
    window_size,
    sinks,
    softmax_scale,
    compressed_lengths=None,
):
    # ── 双模式：按 KV 槽字节数选 quant_mode，按 A3 记录格式选参数集 ──────────
    #  258B → 3（TurboQuant）：旧参数集，A5 / A3+TQ 行为逐字节不变
    #  544B → 1（fp8 / int8 连续量化）；608B → 1（int8_mixed）
    #  A5（VLLM_ASCEND_A3_SOFT_KV8=0）恒走旧参数集，quant_mode 保持 3。
    slot_bytes = int(ori_kv.shape[-1]) if ori_kv is not None else 0
    a3_mode1 = a3_kv_dtype() is not None and kv_slot_format(slot_bytes) != "turboquant"
    quant_mode = kv_slot_quant_mode(slot_bytes) if a3_mode1 else 3

    if a3_mode1:
        # A3 连续量化（quant_mode=1）的参数契约（逐条有源码依据，见 TI/VLLM_PATCH.md §3）：
        #  * ori_sparse_indices / ori_topk_length 不传：原始侧按 BAND mask + 窗口取 KV
        #  * cmp_mask_mode=0 + cmp_ratio=1 + cmp_* 全 None：本阶段只传 ori_kv
        #  * ori_mask_mode=4（BAND）+ ori_win_left=window-1 + ori_win_right=0
        #  * rope_head_dim=64（CANN README:338「仅支持 64」）
        #  注意：long-KV（cmp 侧）本阶段只写不读 —— attention 只看 SWA 窗口，
        #  精度验收必须等读侧 FP8 实例就位后重新做。
        ori_indices = None
        ori_lengths = None
        cmp_kv = None
        cmp_indices = None
        cmp_lengths = None
        cmp_block_table = None
        seqused_cmp_kv = None
        cmp_residual_kv = None
        cmp_mask_mode = 0
        cmp_ratio = 1
    else:
        ori_indices, ori_lengths = _resolve_window_indices(q, metadata, window_size)
        has_cmp = cmp_kv is not None
        cmp_residual_kv = None
        if has_cmp:
            cmp_indices = compressed_indices[:, None, :].to(torch.int32).contiguous()
            cmp_lengths = compressed_lengths
        else:
            # TurboQuant(quant_mode=3) 要求 cmp_sparse_indices 的 K 为 512/1024，
            # 即使没有 compressed KV 也要给一个占位，否则 tiling 直接拒绝。
            if compressed_indices is not None:
                cmp_indices = compressed_indices[:, None, :].to(torch.int32).contiguous()
            else:
                _k = 512  # index_topk；TurboQuant 只接受 512/1024
                cmp_indices = torch.zeros((q.shape[0], 1, _k), dtype=torch.int32, device=q.device)
            cmp_lengths = None
        cmp_block_table = metadata.attention.block_table if has_cmp else None
        seqused_cmp_kv = metadata.attention.cache_seq_lens if has_cmp else None
        cmp_mask_mode = 3
        cmp_ratio = 4
    ori_mask_mode = 4
    ori_win_left = max(0, int(window_size) - 1)
    ori_win_right = 0
    task_metadata = metadata.swa.smla_metadata
    wait_for_device_metadata(DeviceMetadataStage.ATTENTION, id(task_metadata))
    import sys as _sys

    def _d(x):
        if x is None:
            return "None"
        if hasattr(x, "shape"):
            return f"{tuple(x.shape)}/{x.dtype}"
        return f"{type(x).__name__}={x}"

    print(
        "[a3-kv8-qsmla] "
        + " ".join(
            f"{k}={_d(v)}"
            for k, v in (
                ("q", q),
                ("ori_kv", ori_kv),
                ("cmp_kv", cmp_kv),
                ("ori_idx", ori_indices),
                ("cmp_idx", cmp_indices),
                ("ori_bt", metadata.swa.block_table),
                ("cu_q", metadata.swa.query_start_loc),
                ("seq_ori", metadata.swa.seq_lens),
                ("ori_len", ori_lengths),
                ("sinks", sinks),
                ("meta", task_metadata),
                ("mode", a3_mode1),
                ("fmt", kv_slot_format(slot_bytes) if slot_bytes else "?"),
                ("quant_mode", quant_mode),
            )
        ),
        file=_sys.stderr,
        flush=True,
    )

    # ② zero record：读之前把从未写过的零 scale record 变成合法零值记录。
    ori_kv = _a3_kv8_safe_init(ori_kv)
    cmp_kv = _a3_kv8_safe_init(cmp_kv)

    # cache 用 int8 存（A3 的 scatter/copy 支持），但算子只接受 float8_e4m3fn
    if ori_kv is not None and ori_kv.dtype != torch.float8_e4m3fn:
        ori_kv = ori_kv.view(torch.float8_e4m3fn)
    if cmp_kv is not None and cmp_kv.dtype != torch.float8_e4m3fn:
        cmp_kv = cmp_kv.view(torch.float8_e4m3fn)
    output, _ = torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=ori_indices,
        cmp_sparse_indices=cmp_indices,
        ori_block_table=metadata.swa.block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=metadata.swa.query_start_loc,
        seqused_ori_kv=metadata.swa.seq_lens,
        seqused_cmp_kv=seqused_cmp_kv,
        ori_topk_length=ori_lengths,
        cmp_topk_length=cmp_lengths,
        cmp_residual_kv=cmp_residual_kv,
        sinks=sinks.detach().float().contiguous(),
        metadata=task_metadata,
        quant_mode=quant_mode,
        rope_head_dim=64,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_right=ori_win_right,
        ori_win_left=ori_win_left,
        cmp_ratio=cmp_ratio,
        softmax_scale=softmax_scale,
        layout_q="TND",
        layout_kv="PA_BBND",
        topk_value_mode=1,
        return_softmax_lse=False,
    )
    return output


def write_attention_cache(
    cache: torch.Tensor,
    flat_slots: torch.Tensor,
    values: torch.Tensor,
    *,
    kind: str,
) -> None:
    if kind == "cmp":
        cache_arg = cache
        group_size = 16
        quant_mode = "mxfp4_bf16"
    elif kind == "win":
        cache_arg = cache.view(torch.float8_e4m3fn)
        group_size = 32
        quant_mode = "mxfp8_bf16"
    else:
        raise ValueError(f"unsupported A5 cache kind: {kind}")
    torch.ops._C_ascend.kv_compress_epilog_v2(
        cache_arg,
        values.contiguous(),
        flat_slots.contiguous(),
        quant_group_size=group_size,
        quant_mode=quant_mode,
        round_scale=True,
        x_scale=1.0,
    )


def write_index_cache(
    cache: tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
    coordinates: torch.Tensor,
    values: torch.Tensor,
) -> None:
    write_mxfp4_indexer_cache(values, coordinates, cache[0], cache[1])


# ---------------------------------------------------------------------------
# A3 写侧打包器：三种 record，逐字节对齐读侧算子（TW/WRITE_PATH.md §1）
#
#   fp8        544B  [0,448) nope FP8  | [448,512) rope FP8  | [512,544) 16×BF16 scale(g32)
#   int8       608B  [0,128) rope BF16 | [128,576) nope INT8(g64) | [576,590) 7×BF16 scale
#                    | [590,608) 18B pad                    （CANN 官方 INT8 CSA 布局）
#   int8_mixed 544B  [0,448) nope INT8(g128，末组 64 lane) | [448,512) rope FP8(g32)
#                    | [512,544) 4×nope scale + 2×rope scale + 10×1.0
#
# 三条不变量（缺一条读侧就静默读错，且不会报错）：
#   1. 写进 record 的 scale 必须**就是**量化时用的那个 scale（先降到 BF16 再编码）；
#   2. FP8 的 scale 只取 2^k，码按 E4M3 网格做 RNE（与 /opt/fp8probe 的 CAST_RINT 同语义）；
#   3. INT8 是对称量化 amax/127 + RNE（ties-to-even），绝不截断（截断有系统性偏置）。
# ---------------------------------------------------------------------------
_E4M3_MAX = 448.0
_E4M3_MIN_NORMAL = 2.0 ** -6


def _pow2_scale(amax: torch.Tensor) -> torch.Tensor:
    """最小的整数 k，使 amax ≤ 448·2^k（E4M3 满量程；纯位运算，不碰 log/除法）。

    与冻结的 /opt/fp8probe kernel 同一约定（实测 kernel 用 448 而不是 127），
    所以两边产出的码字/scale 可以逐字节相同。
    amax = 2^e·f，f∈[1,2)；448 = 2^9·0.875，所以
      k = e-8（f ≤ 1.75，即尾数 ≤ 0x600000）或 e-7（f > 1.75）。amax = 0 → k = 0。
    """
    amax = amax.to(torch.float32)
    bits = amax.view(torch.int32)
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    k = (exp - 127 - 8) + (mant > 0x600000).to(torch.int32)   # 0.75·2^23 = 0x600000
    return torch.where(amax > 0, k, torch.zeros_like(k))


def _e4m3_encode(y: torch.Tensor) -> torch.Tensor:
    """FP32 → E4M3 码（int32，低 8 位有效），RNE（ties-to-even），纯位运算。

    与 `Cast(..., RoundMode::CAST_RINT)` 同语义：正规数丢低 20 位尾数做 RNE；
    次正规（|y| < 2^-6）走 2^-9 网格（m = rint(|y|·2^9)，m=8 正好衔接 2^-6，
    码 0x08）。|y| ≤ 448 由调用方保证（超出会被 clamp 到 448）。
    """
    y = y.to(torch.float32).contiguous()
    a = y.abs()
    bits = a.view(torch.int32)
    # 正规：RNE 到 3 位尾数（丢 20 位）= 加 half-1 + lsb 后截断。
    # 注意所有 `<<` 都写成整数乘法、`>>31` 写成 signbit：NPU 上 int32 左移单条
    # 1.6 ms（右移 0.13 ms），编码器里两条左移就能把写侧拖到 3.5 ms。
    rb = (bits + 0x7FFFF + (bits & 0x100000 != 0).to(torch.int32)) & ~0xFFFFF
    q = rb >> 20                      # q = 8·e_fp32 + M = 8(e+120) + M
    code_n = q - 960                  # code = 8e + M
    # 次正规：|y| < 2^-6 时 E4M3 的步长是 2^-9
    code_s = torch.round(a * 512.0).to(torch.int32).clamp(0, 8)
    code = torch.where(a >= _E4M3_MIN_NORMAL, code_n, code_s).clamp(0, 0x7F)
    sign = torch.signbit(y).to(torch.int32)
    return code + sign * 128


def _pow2_float(k: torch.Tensor) -> torch.Tensor:
    """k → 2^k（用 fp32 位模式构造，逐位精确；k 已 clamp 到 [-120, 100]）。"""
    return ((127 + k) * 8388608).view(torch.float32)      # << 23 用乘法代替（NPU 左移很慢）


def _bf16_scale(g: torch.Tensor, qmax: float = 127.0) -> torch.Tensor:
    """对称量化 scale = BF16(amax/qmax)，以 fp32 返回（值本身就是 bf16 精确值）。"""
    return (g.abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-12).to(torch.bfloat16).float()


def _int8_code(g: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """RNE 到整数域并 clamp 到 [-127,127]（128 留作对称量的正侧余量）。"""
    return torch.round(g / scale).clamp(-127, 127).to(torch.int8)


def _fp8_544_rows(values: torch.Tensor, slot_bytes: int) -> torch.Tensor:
    """FP8 记录：nope448×E4M3 + rope64×E4M3 + 16×BF16 scale(g32) = 544 B。

    与冻结的 /opt/fp8probe `pack_544` 同一套算法（scale = 2^k、码按 E4M3 RNE），
    但**全在 device 上**（fp8probe 要 H2D/D2H 来回搬）；实测两者码字逐位一致。
    """
    x = values.detach().reshape(-1, 512).to(torch.float32)
    t = x.shape[0]
    if slot_bytes != 544:
        raise RuntimeError(f"fp8 544 record needs a 544 B slot, got {slot_bytes}")
    g = x.reshape(t, 16, 32)
    k = _pow2_scale(g.abs().amax(-1)).clamp(-120, 100)              # [T,16]
    inv = _pow2_float(-k).unsqueeze(-1)                            # 2^-k
    code = _e4m3_encode((g * inv).reshape(t, 512).clamp(-_E4M3_MAX, _E4M3_MAX))
    scale_bf16 = _pow2_float(k).to(torch.bfloat16)
    return torch.cat(
        [
            code.to(torch.uint8).reshape(t, 512),
            scale_bf16.contiguous().view(torch.uint8).reshape(t, 32),
        ],
        dim=1,
    )


def _fp8_544_rows_kernel(values: torch.Tensor, slot_bytes: int) -> torch.Tensor:
    """A/B 参考路径：冻结的软件 FP8 kernel（/opt/fp8probe，实测 REL_ERR 2.22%）。

    默认不走（见 `VLLM_ASCEND_A3_KV_FP8_KERNEL`）。它把张量拉回 CPU 再 H2D/D2H，
    比纯 torch 版本慢一个量级，保留只为可复现对账。
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("fp8probe_py", "/opt/fp8probe/fp8probe_py.py")
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("/opt/fp8probe/fp8probe_py.py not found")
    fp8probe_py = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fp8probe_py)
    y, k = fp8probe_py.quantize(values.detach().reshape(-1, 512))
    packed = fp8probe_py.pack_544(y, k)                            # [T,544] uint8 (numpy, CPU)
    if packed.shape[1] != slot_bytes:
        raise RuntimeError(f"packed slot is {packed.shape[1]} B but cache slot is {slot_bytes}")
    return torch.from_numpy(packed)


def _int8_608_rows(values: torch.Tensor, slot_bytes: int) -> torch.Tensor:
    """CANN 官方 INT8 CSA 记录 = 608 B（读侧 decode_int8_cache 的 ABI）。

    [0,128) rope64×BF16（不量化、逐字节透传）| [128,576) nope448×INT8(g64)
    | [576,590) 7×BF16 scale | [590,608) 18 B pad（全 0）。
    """
    x = values.detach().reshape(-1, 512).to(torch.float32)
    t = x.shape[0]
    if slot_bytes != 608:
        raise RuntimeError(f"int8 608 record needs a 608 B slot, got {slot_bytes}")
    rope_bytes = x[:, 448:512].to(torch.bfloat16).contiguous().view(torch.uint8).reshape(t, 128)
    g = x[:, :448].reshape(t, 7, 64)
    scale = _bf16_scale(g)                                          # [T,7,1]
    code = _int8_code(g, scale).reshape(t, 448).contiguous().view(torch.uint8)
    return torch.cat(
        [
            rope_bytes,
            code,
            scale.to(torch.bfloat16).contiguous().view(torch.uint8).reshape(t, 14),
            torch.zeros((t, 18), dtype=torch.uint8, device=x.device),
        ],
        dim=1,
    )


def _int8_mixed_544_rows(values: torch.Tensor, slot_bytes: int) -> torch.Tensor:
    """INT8 混合记录 = 544 B（读侧 decode_mixed_int8_cache 的 ABI，TB 变体 B）。

    [0,448) nope448×INT8（g128 → 4 个 scale，末组只覆盖 64 lane）|
    [448,512) rope64×FP8 E4M3（g32 → 2 个 2^k scale）|
    [512,544) 16×BF16 scale（slot0..3 = nope、4..5 = rope，其余 10 个填 1.0，读侧不使用）。
    与 fp8 544 同带宽，精度好 3 倍（TB 4 层实测 0.876% vs 2.650%）。
    """
    x = values.detach().reshape(-1, 512).to(torch.float32)
    t = x.shape[0]
    if slot_bytes != 544:
        raise RuntimeError(f"int8_mixed 544 record needs a 544 B slot, got {slot_bytes}")
    nope = x[:, :448]
    padded = torch.cat([nope, nope.new_zeros((t, 64))], dim=1).reshape(t, 4, 128)
    n_scale = _bf16_scale(padded)                                   # [T,4,1]
    n_code = _int8_code(padded, n_scale).reshape(t, 512)[:, :448]
    rope = x[:, 448:512].reshape(t, 2, 32)
    k = _pow2_scale(rope.abs().amax(-1)).clamp(-120, 100)           # [T,2]
    r_code = _e4m3_encode((rope * _pow2_float(-k).unsqueeze(-1)).reshape(t, 64))
    r_scale = _pow2_float(k).to(torch.bfloat16)
    scales = torch.cat([n_scale.squeeze(-1).to(torch.bfloat16), r_scale], dim=1)  # [T,6]
    return torch.cat(
        [
            n_code.contiguous().view(torch.uint8),
            r_code.to(torch.uint8).reshape(t, 64),
            scales.contiguous().view(torch.uint8).reshape(t, 12),
            torch.ones((t, 10), dtype=torch.bfloat16, device=x.device)
            .view(torch.uint8)
            .reshape(t, 20),
        ],
        dim=1,
    )


_A3_WRITE_LOG_BUDGET = int(_os.environ.get("VLLM_ASCEND_A3_KV_LOG_BUDGET", "3"))
_A3_WRITE_LOG_COUNT = {"n": 0}


def _log_a3_write_once(kind, record, cache, values, flat_slots) -> None:
    """写侧每层每 step 都会调，不能刷屏：前 N 次各打一行（N 由 env 控制）。"""
    if _A3_WRITE_LOG_COUNT["n"] >= _A3_WRITE_LOG_BUDGET:
        return
    _A3_WRITE_LOG_COUNT["n"] += 1
    print(
        f"[a3-kv8] write kind={kind} record={record} slot={int(cache.shape[-1])}B "
        f"cache={tuple(cache.shape)}/{cache.dtype} values={tuple(values.shape)}/{values.dtype} "
        f"slots={tuple(flat_slots.shape)} (log budget {_A3_WRITE_LOG_BUDGET})",
        file=_sys.stderr,
        flush=True,
    )


# --- W7: 固定 shape 的 masked row scatter（graph-safe 的 "-1 slot" 跳过）------
# 语义与 A5 kv_compress_epilog_v2 kernel 的跳过完全一致：
#   for row in range(T): s = slots[row]; if 0 <= s < cache_rows: cache[s] = raw[row]
# launch shape = (T,)，只由 token 数决定；哪些行是 -1 不改变任何 shape，
# 因此 eager 与 graph capture 走同一条路径、写出字节相同（验证矩阵见 GRAPH_SLOT.md）。
@triton.jit(do_not_specialize=["NUM_SLOTS", "BLOCK_STRIDE"])
def _scatter_packed_rows_kernel(
    CACHE,
    SLOTS,
    ROWS,
    NUM_SLOTS,
    BLOCK_STRIDE,
    BLOCK_ROWS: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(SLOTS + row).to(tl.int64)
    if (slot >= 0) & (slot < NUM_SLOTS):
        # flat slot = page * BLOCK_ROWS + offset；块内行连续，块间可以有洞
        # （A3 KV8 沿用 bf16 的 block page 131072B，而 544B record 只填 69632B）。
        page = slot // BLOCK_ROWS
        offset = slot % BLOCK_ROWS
        base = page * BLOCK_STRIDE + offset * ROW_STRIDE
        offs = tl.arange(0, BLOCK)
        mask = offs < ROW_BYTES
        val = tl.load(ROWS + row * ROW_BYTES + offs, mask=mask, other=0)
        tl.store(CACHE + base + offs, val, mask=mask)


def _is_capturing() -> bool:
    """当前 stream 是否在 graph capture 里（capture 内禁止 D2H / .item()）。"""
    try:
        return bool(torch.npu.is_current_stream_capturing())
    except Exception:  # pragma: no cover - 老 torch_npu
        return False



# --- ② zero record 修复：读侧 Python 预处理（不改 CANN）------------------------
# vLLM 的 KV cache 由 torch.zeros 分配：从未写过的行是「zero record」（码=0、scale=0）。
# 读侧 arch22 的 FP8/INT8 CSA 输入按 `fp32_bits = ((b&0x80)<<24) + ((b&0x7F)<<20)
# + ((120+k)<<23)` 这类**指数域拼接**反解（b 是码字节、k 由 bf16 scale 的指数字段推），
# scale=0 时指数域退化/回绕 ⇒ 反解出 ~1e38、inf（而不是 0）⇒ attention 里 exp 溢出 ⇒ NaN。
# 这里在**读之前**把「scale 段全零」的 record 改成合法零值记录（码段保持 0、
# scale=bf16 1.0）：反解恒等于 0，数学上就是「这些行没写过 = 零 KV」。
#   * 幂等：每个 cache 张量只扫一次（写过的行 scale 恒非 0，不可能再变回 0）；
#   * 不碰已写行：只有 scale 段**全零**的行才被改写；
#   * **默认关**（实测当前两个 vendor 的 zero record 反解就是 0，见 WRITE_FIX.md §二）；
#     VLLM_ASCEND_A3_KV_SAFE_INIT=1 打开。只认 544/608 record。
_A3_SAFE_INIT_RAW = _os.environ.get("VLLM_ASCEND_A3_KV_SAFE_INIT", "0").strip().lower()
A3_SAFE_INIT_ENABLED = _A3_SAFE_INIT_RAW not in ("0", "off", "false", "no", "disabled")
# record 字节布局 → 读侧 scale 段（bf16 元素偏移, bf16 元素个数）：
#   fp8 544        [512,544) 16×bf16（g32，nope+rope 全用）
#   int8_mixed 544 [512,544) 4×nope + 2×rope + 10×填充（全 1.0 也是合法 record）
#   int8 608       [576,590) 7×bf16（g64）；[590,608) 是 pad，读侧忽略
_A3_SAFE_SCALE_SPAN = {544: (256, 16), 608: (288, 7)}
_A3_SAFE_INIT_DONE = set()


def _a3_kv8_safe_init(cache):
    """把「从未写过」的 zero record 的 scale 段由 0 改成 bf16 1.0（码段保持 0）。"""
    if not A3_SAFE_INIT_ENABLED or not torch.is_tensor(cache) or cache.dim() < 2:
        return cache
    slot_bytes = int(cache.shape[-1])
    span = _A3_SAFE_SCALE_SPAN.get(slot_bytes)
    if span is None or cache.dtype not in (torch.int8, torch.uint8):
        return cache
    if int(cache.stride(-1)) != 1:
        return cache
    key = (int(cache.data_ptr()), tuple(cache.shape), cache.dtype)
    if key in _A3_SAFE_INIT_DONE:
        return cache
    if _is_capturing():
        # capture 内禁止 D2H 统计；预热阶段（eager）已经处理过每个 cache。
        return cache
    off, count = span
    try:
        scales = cache.view(torch.bfloat16).narrow(-1, off, count)
    except RuntimeError:  # 步长不可被 2 整除等极端布局：不动
        return cache
    zero = (scales == 0).all(dim=-1)
    n_zero = int(zero.sum())
    _A3_SAFE_INIT_DONE.add(key)
    if n_zero:
        scales[zero] = 1.0
    print(
        f"[a3-kv8-safe-init] cache={tuple(cache.shape)} stride={tuple(cache.stride())} "
        f"slot={slot_bytes}B scale_bf16[{off}:{off + count}] "
        f"zero_scale_rows={n_zero}/{int(zero.numel())} -> scale=1.0 (code=0 ⇒ KV=0)",
        file=_sys.stderr,
        flush=True,
    )
    return cache


def packed_cache_num_slots(cache: torch.Tensor) -> int:
    """flat slot 空间大小 = blocks × block_rows（4D 池布局）或行数（2D 行表）。"""
    return int(cache.shape[0]) * int(cache.shape[1]) if cache.dim() == 4 else int(cache.shape[0])


def scatter_packed_rows_(cache: torch.Tensor, slots: torch.Tensor, rows: torch.Tensor) -> None:
    """把 rows[i] 逐字节写到 cache 的 flat slot ``slots[i]``；负数/越界行直接跳过。

    **固定 shape**：grid 只由 ``rows.shape[0]`` 决定，负 slot 的个数/位置不改变
    launch、也不产生任何数据相关的 tensor 形状 —— 这是 graph capture 能过的前提。
    跳过语义与 A5 ``kv_compress_epilog_v2`` 的 ``if (slot < 0 || slot >= cacheRows)
    continue;`` 一致。

    寻址按 cache 的**真实步长**做（``page * block_stride + offset * row_stride``），
    所以块间带洞的池布局（A3 KV8：block page 131072B、544B record 只填 69632B）
    也能写对 —— 不能先 ``cache.reshape(-1, slot_bytes)``：非连续时会**拷贝**，
    写进临时张量就丢了（见 GRAPH_SLOT.md §五.0）。
    """
    if slots.dim() != 1:
        slots = slots.reshape(-1)
    n_rows = int(rows.shape[0])
    if int(slots.numel()) != n_rows:
        raise RuntimeError(f"slot/row count mismatch: slots={int(slots.numel())} rows={n_rows}")
    if n_rows == 0:
        return
    row_bytes = int(rows.shape[1])
    if rows.dtype != cache.dtype:
        raise RuntimeError(f"row dtype mismatch: cache={cache.dtype} rows={rows.dtype}")
    if cache.stride(-1) != 1:
        raise RuntimeError(f"cache row must be innermost-contiguous, stride={cache.stride()}")
    if cache.dim() == 4 and int(cache.shape[2]) == 1 and int(cache.shape[1]) > 0:
        # (blocks, block_rows, 1, row_bytes) —— 生产布局（允许块间带洞）
        block_rows = int(cache.shape[1])
        block_stride = int(cache.stride(0))
        row_stride = int(cache.stride(1))
        num_slots = int(cache.shape[0]) * block_rows
    elif cache.dim() == 2:
        # 纯 flat 行表（单测/回退）
        block_rows = int(cache.shape[0])
        block_stride = 0
        row_stride = int(cache.stride(0))
        num_slots = block_rows
    else:
        raise RuntimeError(f"unsupported cache layout {tuple(cache.shape)} / {cache.stride()}")
    if int(cache.shape[-1]) != row_bytes:
        raise RuntimeError(f"row width mismatch: cache={int(cache.shape[-1])} rows={row_bytes}")
    _scatter_packed_rows_kernel[(n_rows,)](
        cache,
        slots,
        rows,
        num_slots,
        block_stride,
        BLOCK_ROWS=block_rows,
        ROW_STRIDE=row_stride,
        ROW_BYTES=row_bytes,
        BLOCK=max(128, triton.next_power_of_2(row_bytes)),
        enable_fp_fusion=False,
    )


def write_attention_cache_a3(
    cache: torch.Tensor,
    flat_slots: torch.Tensor,
    values: torch.Tensor,
    *,
    kind: str,
) -> None:
    """A3 版写侧：把 post-RoPE 的 KV 量化成 544/608 B record 写进 cache。

    记录格式由 ``VLLM_ASCEND_A3_KV_QUANT``（或旧的 SOFT_KV8 + KV_DTYPE）决定，
    并与 cache 槽字节数交叉断言（`cache_config.kv_slot_format`），错配直接报错：

      fp8        544B  nope448×FP8   + rope64×FP8  + 16×BF16 scale(g32)
      int8       608B  rope64×BF16   + nope448×INT8 + 7×BF16 scale(g64) + 18B pad
      int8_mixed 544B  nope448×INT8(g128) + rope64×FP8(g32) + 6×scale + 10×1.0

    ``kind="win"`` 写 SWA（attention 的 kv），``kind="cmp"`` 写 long-KV
    （compressor 的 latent）。两者输入同构：都是 [T,512] 的 post-RoPE 值、
    rope 在最后 64 维（dsa_v41.py:419-425 / :686-688），所以只换数据源，
    量化逻辑完全复用 —— 记录布局对 win/cmp 是同一个。
    """
    if kind not in ("win", "cmp"):
        raise ValueError(f"unsupported A3 cache kind: {kind}")
    slot_bytes = int(cache.shape[-1])

    # TurboQuant(4bit) 路径：直接用官方的 TurboQuantLatent（Hadamard + codebook），
    # 它内部调 cann_ops_nn.turbo_quant，产出 258 B/token 的 packed 布局。
    if _os.environ.get("KV_DTYPE") == "turboquant_4bit_nc":
        if kind == "cmp":
            # TurboQuant 的 long-KV 仍是 A5 的 FP4 320B 记录，写侧只有
            # kv_compress_epilog_v2（A3 无此算子）→ 显式失败，绝不静默写坏 cache。
            raise NotImplementedError(
                "TurboQuant long-KV needs the A5 FP4 epilogue (kv_compress_epilog_v2), "
                "unavailable on A3; set VLLM_ASCEND_A3_KV_QUANT=fp8|int8|int8_mixed"
            )
        from vllm_ascend.quantization.methods.kv_cache.turboquant.latent import (
            TurboQuantLatent,
        )

        _lat = TurboQuantLatent(legacy_hadamard=True)
        # compress() 才是真正的量化（Hadamard 旋转 + codebook + 2 字节 scale），
        # 返回 [T, 1, 258] 的 packed 布局；forward() 只是旋转。
        _out = _lat.compress(values.reshape(-1, 512))
        _rows = _out.reshape(_out.shape[0], -1).view(torch.uint8)
        _slot = cache.shape[-1]
        if _rows.shape[1] != _slot:
            raise RuntimeError(f"turboquant slot {_rows.shape[1]} != cache slot {_slot}")
        _cache_u8 = cache.view(torch.uint8)
        _slots = flat_slots.long()
        if _slots.device != _cache_u8.device:
            _slots = _slots.to(_cache_u8.device)
        # 与上面同样的固定 shape 跳过（turboquant 的 258B record 也是 8bit 行）。
        scatter_packed_rows_(_cache_u8, _slots, _rows.to(cache.device).contiguous().view(torch.uint8))
        return

    record = kv_slot_format(slot_bytes)
    _log_a3_write_once(kind, record, cache, values, flat_slots)

    if record == A3_KV_DTYPE_FP8:
        if _os.environ.get("VLLM_ASCEND_A3_KV_FP8_KERNEL") == "1":
            rows = _fp8_544_rows_kernel(values, slot_bytes)
        else:
            rows = _fp8_544_rows(values, slot_bytes)
    elif record == A3_KV_DTYPE_INT8:
        rows = _int8_608_rows(values, slot_bytes)
    elif record == A3_KV_DTYPE_INT8_MIXED:
        rows = _int8_mixed_544_rows(values, slot_bytes)
    else:  # pragma: no cover - kv_slot_format 已穷举
        raise RuntimeError(f"unsupported A3 KV record: {record}")

    # 注意：**不要** cache.reshape(-1, slot_bytes)。A3 KV8 的池布局块间带洞
    # （block page 131072 B，而 544 B record 只填 69632 B），reshape 会拷贝，
    # 写进临时张量就静默丢了（GRAPH_SLOT.md 五.0）。掩码 scatter 直接按 cache
    # 的真实步长寻址。
    raw = rows.to(device=cache.device).contiguous()
    if raw.dtype != cache.dtype:
        # A3 的 cache 是 torch.int8（NPU 上 index_copy_ 对 uint8 支持不稳），
        # 这里是**逐位重解释**，不是数值 cast（128 不能变成 127）。
        if cache.dtype != torch.int8:
            raise RuntimeError(f"unsupported A3 cache dtype {cache.dtype}")
        raw = raw.view(torch.int8)
    slots = flat_slots.long()
    if slots.device != cache.device:
        # 引擎里 flat_slot_mapping 在 device 上；CPU 侧单测传 CPU 索引，这里兜一下。
        slots = slots.to(cache.device)
    # PAD 行（graph 固定 carrier 的尾行）和 ratio-2 未凑满压缩组的行都会被
    # `build_packed_cache_slot_mapping` 写成 -1（``tl.where(valid, physical, -1)``）。
    # A5 原生 write_attention_cache（kv_compress_epilog_v2 kernel）在 kernel 内部
    # ``if (slot < 0 || slot >= cacheRows) continue;`` 跳过；A3 原来用
    # ``flat.index_copy_``，在 NPU 上走 AICPU ScatterUpdate —— 负索引/越界索引直接
    # 507018 abort 整个进程（最小复现见 W1/t_ic3.py）。先 ``keep = slots >= 0``
    # gather 掉无效行虽然能跑，但那是**数据相关的 shape**，graph capture 拒收
    # （这就是 W1 只能 eager 的原因）。改用固定 shape 的 masked row scatter：
    # launch 只看行数，跳过发生在 kernel 内部。
    scatter_packed_rows_(cache, slots, raw)




# 必须显式 import 才会把 torch.ops.cann_ops_transformer.* 注册进来
try:  # pragma: no cover
    import cann_ops_transformer  # noqa: F401
except ImportError:  # pragma: no cover
    pass


class MixedQuantPackedCacheOpsA3:
    """A3 版 packed-cache ABI：读侧复用 A5 的实现，只替换写侧。"""

    build_packed_cache_slot_mapping = staticmethod(build_packed_cache_slot_mapping)
    build_window_indices = staticmethod(build_window_indices_triton)
    build_smla_metadata = staticmethod(build_smla_metadata)
    qsmla = staticmethod(qsmla)
    run_a5_indexer = staticmethod(run_a5_indexer)
    write_attention_cache = staticmethod(write_attention_cache_a3)
    write_index_cache = staticmethod(write_index_cache)
    # A3 没有 net native wheel 提供的顶层 `ops` 包（packaged_attention 里的
    # import_module(f"ops.{name}") 会 ModuleNotFoundError），改用 torch.ops 注册名。
    mixed_quant_sparse_flash_mla_metadata = staticmethod(_smla_metadata_a3)
    quant_lightning_indexer_metadata = staticmethod(_qli_metadata_a3)


class MixedQuantPackedCacheOps:
    """Packed mixed-quant cache ABI selected by the hardware adaptor."""

    build_packed_cache_slot_mapping = staticmethod(build_packed_cache_slot_mapping)
    build_window_indices = staticmethod(build_window_indices_triton)
    build_smla_metadata = staticmethod(build_smla_metadata)
    qsmla = staticmethod(qsmla)
    run_a5_indexer = staticmethod(run_a5_indexer)
    write_attention_cache = staticmethod(write_attention_cache)
    write_index_cache = staticmethod(write_index_cache)
    mixed_quant_sparse_flash_mla_metadata = staticmethod(ops.mixed_quant_sparse_flash_mla_metadata)
    quant_lightning_indexer_metadata = staticmethod(ops.quant_lightning_indexer_metadata)
