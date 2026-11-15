# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Framework-side V4.1 layer-outermost cache placement and allocation."""

from dataclasses import dataclass, replace
from itertools import zip_longest

import torch
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import may_override_num_blocks
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    declared_kwarg,
)
from vllm_ascend.device.hardware_profile import (
    HardwareCapability,
    get_current_hardware_profile,
)
from vllm_ascend.utils import vllm_version_is

# vLLM #61183 moved `may_override_num_blocks` to `kv_cache_planning` on main;
# the v0.31.0 lane still ships it in `kv_cache_utils`.
if vllm_version_is("0.31.0"):
    from vllm.v1.core.kv_cache_utils import may_override_num_blocks
else:
    from vllm.v1.core.kv_cache_planning import may_override_num_blocks

STATE_RING_ROWS = 32

# ---------------------------------------------------------------------------
# ring 摘出 block pool（VLLM_ASCEND_RING_OUT_OF_POOL，默认关）
#
# 现状：FP32 compressor state ring 是 slot 内的一个 alias，slot 页 =
# max(long-KV+indexer, ring, SWA)。ring 每请求只要 1 页
# （CircularBufferSpec.max_num_blocks_per_req == 1，上游 kv_cache_interface.py:909），
# 却把 num_blocks 个页的地板都抬到 STATE_RING_ROWS*2*head_dim*4 = 131,072 B。
#
# gate 打开后：ring 不再是任何 slot 的 alias，改用独立 tensor
# （max_num_seqs+1 页，自带 BlockPool，与主池 id 空间分离）；slot 页回落到
# max(long-KV+indexer, SWA)。ring 仍是 FP32、仍是 32 行 × 2*head_dim，精度零损失。
# ---------------------------------------------------------------------------
RING_OUT_OF_POOL_ENV = "VLLM_ASCEND_RING_OUT_OF_POOL"


def ring_out_of_pool() -> bool:
    """是否把 V4.1 compressor state ring 从 block pool 里摘出去（默认关）。"""
    import os as _os

    return _os.environ.get(RING_OUT_OF_POOL_ENV) == "1"


def is_ring_state_spec(spec) -> bool:
    """是否是「被摘出池」的 ring spec（gate 关闭时恒 False）。

    V4.1 的 ``*.compressor.state_cache`` 是本仓库唯一使用 CircularBufferSpec
    的 cache（compressor.py:36-47），且只有在 gate 打开时才把它当作 ring。
    """
    return ring_out_of_pool() and isinstance(spec, CircularBufferSpec)


def ring_transfer_enabled(vllm_config: VllmConfig | None) -> bool:
    """PD/CED 部署下 ring 的 compressor state 必须随 KV 一起转移。

    T-RING 把 ring 摘成每请求私有页后，把 ring group 标成不可外部传输
    （``enable_kv_transfer=False``）。非 PD 部署下这是对的：ring 是每请求
    scratch，不占转移列表。但 PD/CED 下 P 的 compressor state 是 D 端第一个
    decode step 的输入——不转移时 D 拿到空/陈旧 ring，``start_pos`` 为奇数
    时 ``residual`` 分支会去读一个不属于本请求的 pooled 行，静默算错
    （TRV/RING_ZEROING.md §4）。

    判定只看 ``kv_transfer_config`` 是否存在且 ``kv_role`` 非空（PD 的
    ``kv_producer`` / ``kv_consumer`` / 单实例 ``kv_both`` 都算）。没有 PD
    角色时返回 False，调用方（``make_cache_groups``）的输出与改动前
    **逐字节一致**。
    """
    if vllm_config is None:
        return False
    transfer = getattr(vllm_config, "kv_transfer_config", None)
    if transfer is None:
        return False
    return bool(getattr(transfer, "kv_role", None))


def ring_pool_num_blocks(vllm_config: VllmConfig) -> int:
    """ring 独立池的页数 = 并发请求数 + 1 个 null 页。

    compressor kernel 用 ``cache_row > 0`` 把 block id 0 当 null
    （compressor_triton.py:251,354），所以可用 id 是 1..N-1；而
    ``scheduler_config.max_num_seqs`` 就是同时在跑的请求数上限
    （上游 gpu_model_runner.py:529 ``self.max_num_reqs = max_num_seqs``），
    每请求恰好 hold 1 页。
    """
    max_num_seqs = int(getattr(vllm_config.scheduler_config, "max_num_seqs", 0) or 0)
    if max_num_seqs <= 0:
        raise ValueError("ring out of pool requires scheduler_config.max_num_seqs > 0")
    return max_num_seqs + 1

# Fixed dimensions accepted by the packaged mixed-quant attention ABI.
# The actual head sizes come from the model config; these bounds reject
# unsupported layouts before allocating caches, rather than override config.
MLA_HEAD_DIM = 512
INDEX_HEAD_DIM = 128
INDEX_FOLD_ROWS = 8
KV_ADDRESS_SPACE_BYTES = 1 << 32


@dataclass(frozen=True)
class CachePlacement:
    name: str
    offset: int
    page_size: int


@dataclass(frozen=True)
class CacheSlot:
    page_size: int
    placements: tuple[CachePlacement, ...]


def uses_a5_packed_cache() -> bool:
    """Select the V4.1 cache ABI from the detected hardware profile.

    A3 (ascend910_93) has no DSV41_PACKED_CACHE capability bit, but the software
    quantization kernel (aclnnFp8Probe) can produce exactly the same 544-byte
    per-token layout, so allow opting into this path explicitly.
    """
    import os as _os

    if _os.environ.get("VLLM_ASCEND_A3_SOFT_KV8") == "1":
        return True
    return get_current_hardware_profile().supports(HardwareCapability.DSV41_PACKED_CACHE)


A3_KV8_INDEX_ENV = "VLLM_ASCEND_A3_KV8_INDEX"


def a3_kv8_index_unpacked() -> bool:
    """A3 软 KV8 下 indexer/QLI 是否退回系统非 packed 路径（默认：是）。

    KV8 的 544B record 是 **attention（SMLA）侧** cache；indexer 的 K cache 与 QLI
    是另一套 cache/算子，历史上只是被 ``uses_a5_packed_cache()`` 顺带一起打开。
    A3 没有 A5 net-native DSL 的 mxfp4 QLI ABI：系统 ``aclnnQuantLightningIndexerV2``
    在 mode=1 下只吃 int8 q / int8 k / fp16 w / fp16 descale，喂 mxfp4 query 直接
    EZ1009。因此 A3 上把 QLI 侧解绑回基线路径，KV8 记录本身不受影响。

    * 未开 A3 软 KV8 → False（A5 与关闭态行为完全不变）
    * A5 原生 packed（有 DSV41_PACKED_CACHE 能力位）→ False（保持 A5 原路径）
    * ``VLLM_ASCEND_A3_KV8_INDEX=packed`` 可恢复旧的「KV8 顺带打开 packed QLI」行为
      （复现 EZ1009 用），其它取值/未设为解绑。
    """
    import os as _os

    if not is_a3_soft_kv8():
        return False
    if get_current_hardware_profile().supports(HardwareCapability.DSV41_PACKED_CACHE):
        return False
    raw = (_os.environ.get(A3_KV8_INDEX_ENV) or "system").strip().lower()
    return raw not in ("packed", "a5", "0", "off", "false", "disabled")


# ---------------------------------------------------------------------------
# A3 软 KV8 记录格式
#
# 开关（两层；`VLLM_ASCEND_A3_KV_QUANT` 是推荐入口，旧的 SOFT_KV8+KV_DTYPE 保留）：
#   VLLM_ASCEND_A3_KV_QUANT = off | fp8 | int8 | int8_mixed
#   VLLM_ASCEND_A3_SOFT_KV8 = 1  +  VLLM_ASCEND_A3_KV_DTYPE = fp8|int8|int8_mixed
#
# 三种物理记录（跨团队 ABI，逐字节对齐读侧算子 decode_*_cache）：
#   fp8        544B  nope448×FP8  + rope64×FP8  + 16×BF16 scale(g32)，无 pad
#   int8       608B  rope64×BF16(128B) + nope448×INT8 + 7×BF16 scale(g64) + 18B pad
#                    （CANN 官方 INT8 CSA 布局，读侧 decode_int8_cache）
#   int8_mixed 544B  nope448×INT8(g128,4 scale) + rope64×FP8(g32,2 scale)
#                    + 16×BF16 scale（前 6 有效，余 10 填 1.0）
# 544 与 608 都只说明「码宽 1 字节」，区分不了 fp8/int8 → 只靠槽字节数判
# quant_mode，格式必须由 env 声明并用槽字节数做自洽断言。
# ---------------------------------------------------------------------------
MLA_ROPE_HEAD_DIM = 64
A3_KV_QUANT_ENV = "VLLM_ASCEND_A3_KV_QUANT"
A3_KV_DTYPE_ENV = "VLLM_ASCEND_A3_KV_DTYPE"
A3_KV_DTYPE_FP8 = "fp8"
A3_KV_DTYPE_INT8 = "int8"
A3_KV_DTYPE_INT8_MIXED = "int8_mixed"
# 读侧报告里的别名（int8_608 = 官方 608 布局；int8_544 = TB 变体 B）
_A3_KV_DTYPE_ALIASES = {"int8_608": A3_KV_DTYPE_INT8, "int8_544": A3_KV_DTYPE_INT8_MIXED}
A3_KV_DTYPES = (A3_KV_DTYPE_FP8, A3_KV_DTYPE_INT8, A3_KV_DTYPE_INT8_MIXED)
A3_TQ_SLOT = 258
A3_SLOT_544 = 544
A3_SLOT_608 = 608
# 兼容旧名（TI 轮的命名，语义已随本轮 ABI 对齐而改变，只保留别名不保留语义）
A3_SLOT_G32 = A3_SLOT_544
A3_SLOT_INT8_MIXED = A3_SLOT_608
_A3_RECORD_BYTES = {
    A3_KV_DTYPE_FP8: A3_SLOT_544,
    A3_KV_DTYPE_INT8: A3_SLOT_608,
    A3_KV_DTYPE_INT8_MIXED: A3_SLOT_544,
}
_A3_RECORD_LOGGED: set[str] = set()


def a3_kv_quant() -> str | None:
    """`VLLM_ASCEND_A3_KV_QUANT` 的规范化取值；None = 未设置（退回旧的两个 env）。

    显式设置时它是唯一真源：`off` 关掉 A3 软 KV8，`fp8|int8|int8_mixed` 既开
    总开关又选记录格式。未设置时行为与本轮改动前逐字节一致。
    """
    import os as _os

    raw = _os.environ.get(A3_KV_QUANT_ENV)
    if raw is None or not raw.strip():
        return None
    value = raw.strip().lower()
    if value in ("0", "off", "none", "false", "disabled"):
        return "off"
    value = _A3_KV_DTYPE_ALIASES.get(value, value)
    if value not in A3_KV_DTYPES:
        raise ValueError(
            f"{A3_KV_QUANT_ENV} must be one of ('off', *{A3_KV_DTYPES}), got {raw!r}"
        )
    return value


def is_a3_soft_kv8() -> bool:
    """A3 软件 KV8 是否打开（VLLM_ASCEND_A3_KV_QUANT≠off 或 SOFT_KV8=1）。"""
    import os as _os

    quant = a3_kv_quant()
    if quant is not None:
        return quant != "off"
    return _os.environ.get("VLLM_ASCEND_A3_SOFT_KV8") == "1"


def a3_kv_dtype() -> str | None:
    """A3 软 KV8 下的记录格式名；None = 保持 A5 packed / TurboQuant 原行为。

    * 未开 A3 软 KV8 → None（A5 与关闭态行为完全不变）
    * KV_DTYPE=turboquant_4bit_nc → None（TurboQuant 的 SWA 是 258B、long-KV 是
      A5 的 FP4 320B，A3 上没有 FP4 写侧，见 write_attention_cache_a3）
    * 格式优先级：VLLM_ASCEND_A3_KV_QUANT > VLLM_ASCEND_A3_KV_DTYPE > 默认 fp8
    """
    import os as _os

    if not is_a3_soft_kv8():
        return None
    if _os.environ.get("KV_DTYPE") == "turboquant_4bit_nc":
        return None
    quant = a3_kv_quant()
    if quant in (None, "off"):
        raw = _os.environ.get(A3_KV_DTYPE_ENV) or A3_KV_DTYPE_FP8
    else:
        raw = quant
    value = _A3_KV_DTYPE_ALIASES.get(raw.strip().lower(), raw.strip().lower())
    if value not in A3_KV_DTYPES:
        raise ValueError(f"{A3_KV_DTYPE_ENV} must be one of {A3_KV_DTYPES}, got {value!r}")
    return value


def a3_record_bytes(head_size: int, kv_dtype: str | None = None) -> int:
    """一条 KV 记录的字节数（A3: 544/608；None 时退回 A5 packed 的 FP4 320）。"""
    kv_dtype = a3_kv_dtype() if kv_dtype is None else kv_dtype
    if kv_dtype is None:
        # A5 packed ABI：两个 FP4 装一个字节 + 每 16 个值一个 BF16 scale
        return head_size // 2 + (head_size // 16) * torch.bfloat16.itemsize
    nope = head_size - MLA_ROPE_HEAD_DIM
    if kv_dtype == A3_KV_DTYPE_FP8:
        # 全部量化成 1 字节 + 每 32 个值一个 BF16 scale
        return head_size + (head_size // 32) * torch.bfloat16.itemsize          # 512 -> 544
    if kv_dtype == A3_KV_DTYPE_INT8:
        # CANN 官方 INT8 CSA：rope 保持 BF16（多 64B）、nope 每 64 值一个 BF16 scale、18B pad
        return head_size + MLA_ROPE_HEAD_DIM + (nope // 64) * torch.bfloat16.itemsize + 18  # 512 -> 608
    # int8_mixed：nope 与 rope 都是 1 字节码，scale 区固定 16 个 BF16
    return head_size + (head_size // 32) * torch.bfloat16.itemsize              # 512 -> 544


def kv_slot_quant_mode(slot_bytes: int) -> int:
    """槽字节数 → 算子 quant_mode：258→3（TurboQuant），544/608→1（连续量化）。"""
    if slot_bytes == A3_TQ_SLOT:
        return 3
    if slot_bytes in (A3_SLOT_544, A3_SLOT_608):
        return 1
    raise RuntimeError(
        f"unsupported KV cache slot {slot_bytes} B; expected {A3_TQ_SLOT} (turboquant), "
        f"{A3_SLOT_544} (fp8/int8_mixed) or {A3_SLOT_608} (int8)"
    )


def kv_slot_format(slot_bytes: int) -> str:
    """槽字节数 + env → 记录格式，并交叉断言（防止 cache 布局与 env 静默错配）。"""
    if slot_bytes == A3_TQ_SLOT:
        return "turboquant"
    kv_dtype = a3_kv_dtype()
    if slot_bytes == A3_SLOT_608:
        if kv_dtype not in (None, A3_KV_DTYPE_INT8):
            raise RuntimeError(
                f"{A3_KV_DTYPE_ENV}={kv_dtype!r} but this cache slot is {slot_bytes} B "
                f"(int8 608); use {A3_KV_QUANT_ENV}=int8"
            )
        return A3_KV_DTYPE_INT8
    if slot_bytes == A3_SLOT_544:
        # 544B 的 fp8 与 int8_mixed 码宽相同，只能信 env；A5 无 A3 env 时 544 即 FP8。
        if kv_dtype == A3_KV_DTYPE_INT8:
            raise RuntimeError(
                f"{A3_KV_DTYPE_ENV}='int8' expects a {A3_SLOT_608} B slot, got {slot_bytes} B"
            )
        if kv_dtype == A3_KV_DTYPE_INT8_MIXED:
            return A3_KV_DTYPE_INT8_MIXED
        return A3_KV_DTYPE_FP8
    raise RuntimeError(f"unsupported KV cache slot {slot_bytes} B")


def make_swa_cache_spec(*, block_size, window_size, head_size, dtype, cache_dtype, bounded_replay=False):
    if uses_a5_packed_cache():
        if head_size != MLA_HEAD_DIM:
            raise ValueError(f"A5 DeepSeek V4.1 requires head size {MLA_HEAD_DIM}, got {head_size}")
        import os as _os

        _a3 = is_a3_soft_kv8()
        _tq = _os.environ.get("KV_DTYPE") == "turboquant_4bit_nc"
        _kv_dtype = a3_kv_dtype()
        if _a3 and _tq:
            # TurboQuant 4bit NC cache：4bit × 512 = 256 字节 + 2 字节 scale。
            # 官方要求 cache.dtype == torch.uint8（写侧再 view 成 int8 走 scatter）。
            head_size = head_size // 2 + 2
            dtype = torch.uint8
            cache_dtype = "turboquant_4bit_nc"
        elif _kv_dtype is not None:
            # A3 软 KV8：SWA 与 long-KV 用同一种记录（fp8/int8_mixed 544B、int8 608B），
            # 否则两条 cache 的格式不一致（读侧只能按一种布局反量化）。
            head_size = a3_record_bytes(head_size, _kv_dtype)
            dtype = torch.int8
            # fp8 沿用历史 cache_dtype 字符串，避免改变既有 A3 路径的 spec 身份
            cache_dtype = {
                A3_KV_DTYPE_FP8: "a3_soft_mxfp8_bf16_scale",                 # 544B
                A3_KV_DTYPE_INT8: "a3_soft_int8_g64_608_bf16_scale",         # 608B
                A3_KV_DTYPE_INT8_MIXED: "a3_soft_int8_mixed_544_bf16_scale", # 544B
            }[_kv_dtype]
        else:
            # FP8 values followed by one BF16 scale per 32 values.
            head_size += (head_size // 32) * torch.bfloat16.itemsize
            if _a3:
                dtype = torch.int8
                cache_dtype = "a3_soft_mxfp8_bf16_scale"
            else:
                dtype = torch.uint8
                cache_dtype = "a5_mxfp8_bf16_scale"
    # TurboQuant 的完整实现在 dsa_v1（旧 backend）里，dsa_v41 还没有移植。
    # is_deepseek_v41_cache() 只看 spec.model_version，所以这里改掉它就能
    # 让 A3 上的 TurboQuant 走 dsa_v1 路径。
    import os as _os

    _mv = "deepseek_v4" if _os.environ.get("KV_DTYPE") == "turboquant_4bit_nc" else "deepseek_v41"
    return AscendSlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        sliding_window=window_size,
        cache_dtype_str=cache_dtype,
        model_version=_mv,
        alignment=None,
        **declared_kwarg(AscendSlidingWindowMLASpec, "bounded_replay", bounded_replay),
    )


def make_mla_cache_spec(*, block_size, head_size, compress_ratio):
    if uses_a5_packed_cache():
        if head_size != MLA_HEAD_DIM:
            raise ValueError(f"A5 DeepSeek V4.1 requires head size {MLA_HEAD_DIM}, got {head_size}")
        kv_dtype = a3_kv_dtype()
        # A3 软 KV8：long-KV 与 SWA 同格式（544B fp8/int8 或 608B int8_mixed），
        # 不再是 A5 的 FP4 320B —— 否则两条 cache 的位宽/布局不一致，读侧无法统一反量化。
        head_size = a3_record_bytes(head_size, kv_dtype)
        if kv_dtype is None:
            dtype = torch.uint8
        else:
            expected = _A3_RECORD_BYTES[kv_dtype]
            if head_size != expected:
                raise ValueError(f"A3 KV8 record for {kv_dtype} must be {expected} B, got {head_size}")
            dtype = torch.int8
            if kv_dtype not in _A3_RECORD_LOGGED:
                _A3_RECORD_LOGGED.add(kv_dtype)
                logger.info("A3 soft KV8 record: %s, %d B/row (long-KV and SWA)", kv_dtype, head_size)
        scale_dtype = torch.bfloat16
    else:
        dtype = torch.bfloat16
        scale_dtype = torch.int8
    return AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        tokens_per_state=compress_ratio,
        model_version="deepseek_v41",
        # vLLM main (#57169) dropped MLA `storage_block_size`; v0.31.0 declares it.
        **({"storage_block_size": block_size // compress_ratio} if vllm_version_is("0.31.0") else {}),
        scale_dtype=scale_dtype,
    )


def make_index_cache_spec(*, block_size, head_size, compress_ratio):
    # 解绑后 indexer 走系统 QLI：int8 K + fp16 descale（与 A3 基线逐字节一致），
    # 不再用 mxfp4 的 uint8 head_size/2 + uint8 scale。
    if uses_a5_packed_cache() and not a3_kv8_index_unpacked():
        if head_size != INDEX_HEAD_DIM:
            raise ValueError(f"A5 DeepSeek V4.1 requires head size {INDEX_HEAD_DIM}, got {head_size}")
        scale_dim = head_size // 32
        head_size //= 2
        dtype = torch.uint8
        scale_dtype = torch.uint8
    else:
        dtype = torch.int8
        scale_dim = 1
        scale_dtype = torch.float16
    return AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        tokens_per_state=compress_ratio,
        model_version="deepseek_v41",
        # vLLM main (#57169) dropped MLA `storage_block_size`; v0.31.0 declares it.
        **({"storage_block_size": block_size // compress_ratio} if vllm_version_is("0.31.0") else {}),
        scale_dim=scale_dim,
        scale_dtype=scale_dtype,
    )


def make_folded_index_cache_spec(*, block_size):
    """QSLI candidate-source view: eight K/scale rows per 544-byte row."""
    if block_size % INDEX_FOLD_ROWS:
        raise ValueError("A5 folded index page must contain full 8-token groups")
    return AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=INDEX_FOLD_ROWS * (INDEX_HEAD_DIM // 2 + INDEX_HEAD_DIM // 32),
        dtype=torch.uint8,
        tokens_per_state=INDEX_FOLD_ROWS,
        model_version="deepseek_v41",
        # vLLM main (#57169) dropped MLA `storage_block_size`; v0.31.0 declares it.
        **({"storage_block_size": block_size // INDEX_FOLD_ROWS} if vllm_version_is("0.31.0") else {}),
    )


def is_deepseek_v41_cache(specs_or_groups):
    if isinstance(specs_or_groups, dict):
        specs = list(specs_or_groups.values())
    else:
        specs = []
        for item in specs_or_groups:
            spec = getattr(item, "kv_cache_spec", item)
            if isinstance(spec, UniformTypeKVCacheSpecs):
                specs.extend(spec.kv_cache_specs.values())
            else:
                specs.append(spec)
    return any(getattr(spec, "model_version", None) == "deepseek_v41" for spec in specs)


def _layer_number(name):
    return int(name.rsplit(".layers.", 1)[1].split(".", 1)[0])


def _draft_layer_number(name):
    return int(("." + name).rsplit(".mtp.", 1)[1].split(".", 1)[0])


def get_layer_tuples(specs):
    """Return DSV4-style ordered layer tuples and their physical page sizes."""
    mla = {name for name, spec in specs.items() if isinstance(spec, AscendMLAAttentionSpec)}
    state = sorted((name for name, spec in specs.items() if isinstance(spec, CircularBufferSpec)), key=_layer_number)
    # ring 摘出池后不再是 slot alias：不再抬 slot 页地板，也不再占 block id。
    # gate 是全局的，所以这里会把所有 ring 一起摘掉（不会造成 slot 序号错位）。
    state = [name for name in state if not is_ring_state_spec(specs[name])]
    swa = {name for name, spec in specs.items() if isinstance(spec, AscendSlidingWindowMLASpec)}

    full = sorted((name for name in mla if name.endswith(".long_kv_cache")), key=_layer_number)
    target_swa = sorted((name for name in swa if ".mtp." not in f".{name}"), key=_layer_number)
    draft_swa = sorted((name for name in swa if ".mtp." in f".{name}"), key=_draft_layer_number)

    # A reduced backbone may contain only the initial SWA layers. Without
    # full-context source slots, each SWA layer needs its own physical slot.
    if not full and len(swa) == len(specs):
        names = target_swa + draft_swa
        return [specs[name].unpadded_page_size_bytes for name in names], [(name,) for name in names]

    layer_tuples: list[tuple[str, ...]] = []
    page_sizes: list[int] = []
    for slot_idx, kv_name in enumerate(full):
        prefix = kv_name.rsplit(".", 1)[0]
        index_name = prefix + ".indexer.k_cache"
        folded_name = index_name + "_folded"
        folded_spec = specs.get(folded_name)
        index_spec = specs[index_name]
        kv_spec = specs[kv_name]
        aliases = ([state[slot_idx]] if slot_idx < len(state) else []) + target_swa[slot_idx :: len(full)]
        kv_bytes = kv_spec.unpadded_page_size_bytes
        index_bytes = index_spec.unpadded_page_size_bytes
        folded_bytes = folded_spec.unpadded_page_size_bytes if folded_spec is not None else 0
        if slot_idx < len(draft_swa):
            aliases.append(draft_swa[slot_idx])
        capacity = max(
            kv_bytes + index_bytes + folded_bytes,
            *(specs[name].unpadded_page_size_bytes for name in aliases),
        )
        layer_tuples.append((kv_name, index_name, *((folded_name,) if folded_spec is not None else ()), *aliases))
        page_sizes.append(capacity)
    return page_sizes, layer_tuples


def plan_cache_slots(specs, *, repack=None):
    """Place index planes in unused source-slot space without changing groups.

    Padded aliases encode the physical slot stride. A stride smaller than
    the original KV+index tuple identifies a repacked group after transport
    to a worker; no process-local configuration or side table is required.
    Folded A5 index resources retain their existing ABI.
    """
    page_sizes, tuples = get_layer_tuples(specs)
    if all(len(names) == 1 for names in tuples):
        return tuple(CacheSlot(size, (CachePlacement(names[0], 0, size),)) for size, names in zip(page_sizes, tuples))
    plans = []
    for size, names in zip(page_sizes, tuples):
        kv_name, index_name, *aliases = names
        folded_name = index_name + "_folded"
        folded = folded_name in aliases
        if folded:
            aliases.remove(folded_name)
        kv_size = specs[kv_name].unpadded_page_size_bytes
        index_size = specs[index_name].unpadded_page_size_bytes
        folded_size = specs[folded_name].unpadded_page_size_bytes if folded else 0
        alias_size = max((specs[name].unpadded_page_size_bytes for name in aliases), default=0)
        plans.append((kv_name, index_name, aliases, kv_size, index_size, folded_name, folded_size, alias_size))
    if repack is None:
        repack = any(
            specs[name].page_size_padded is not None and specs[name].page_size_padded < kv_size + index_size
            for _, _, aliases, kv_size, index_size, _, _, _ in plans
            for name in aliases
        )
    used = [plan[3] + plan[4] + plan[6] for plan in plans]
    moved = {}
    receivers = {}
    if repack and not any(plan[6] for plan in plans):
        for i, (_, _, aliases, kv_size, index_size, _, _, alias_size) in enumerate(plans):
            if not aliases or i in receivers or kv_size + index_size <= max(kv_size, alias_size):
                continue
            for j, size in enumerate(page_sizes):
                if j == i or j in moved or j in receivers:
                    continue
                if size - used[j] >= index_size:
                    moved[i] = (j, used[j])
                    receivers[j] = i
                    used[j] += index_size
                    break
    slots = []
    for i, (kv_name, index_name, aliases, kv_size, index_size, folded_name, folded_size, alias_size) in enumerate(
        plans
    ):
        size = max(kv_size, alias_size) if i in moved else page_sizes[i]
        placements = [CachePlacement(kv_name, 0, kv_size)]
        if i not in moved:
            # The padding after a resident index can hold a relocated index;
            # its logical payload size does not change.
            index_page = index_size if i in receivers or folded_size else size - kv_size
            placements.append(CachePlacement(index_name, kv_size, index_page))
        if folded_size:
            placements.append(CachePlacement(folded_name, kv_size + index_size, size - kv_size - index_size))
        placements.extend(CachePlacement(name, 0, size) for name in aliases)
        if i in receivers:
            source = receivers[i]
            placements.append(CachePlacement(plans[source][1], moved[source][1], plans[source][4]))
        if any(p.offset < 0 or p.offset + p.page_size > size for p in placements):
            raise ValueError("DeepSeek V4.1 cache plane exceeds its physical slot")
        slots.append(CacheSlot(size, tuple(placements)))
    names = [p.name for slot in slots for p in slot.placements]
    # 被摘出池的 ring 不占 slot，但仍由独立 KVCacheTensor 覆盖
    # （见 get_deepseek_v41_kv_cache_config）。
    expected = {name for name, spec in specs.items() if not is_ring_state_spec(spec)}
    if len(names) != len(set(names)) or set(names) != expected:
        raise ValueError("DeepSeek V4.1 slot placement must cover every resource exactly once")
    return tuple(slots)


def group_cache_specs(specs, *, repack=False):
    """Merge full-context resources and pad layer tuples without mutating inputs."""
    if specs and all(isinstance(spec, AscendSlidingWindowMLASpec) for spec in specs.values()):
        return [UniformTypeKVCacheSpecs.from_specs({name: spec}) for name, spec in specs.items()]
    _, layer_tuples = get_layer_tuples(specs)
    padded = {
        placement.name: replace(specs[placement.name], page_size_padded=placement.page_size)
        for slot in plan_cache_slots(specs, repack=repack)
        for placement in slot.placements
    }

    mla_names = [name for name, spec in padded.items() if isinstance(spec, AscendMLAAttentionSpec)]
    state_names = [name for name, spec in padded.items() if isinstance(spec, CircularBufferSpec)]
    # ring 摘出池时不进 padded（不占 slot），但仍必须留在自己的 group 里，
    # 否则调度器不会为它分配「每请求 1 页」的 block id。保持原有顺序追加。
    state_names = state_names + [
        name
        for name, spec in specs.items()
        if isinstance(spec, CircularBufferSpec) and name not in padded
    ]
    # gate 关闭时 padded 里就有全部 ring，行为与改动前逐字节一致。
    state_specs = {name: padded[name] if name in padded else specs[name] for name in state_names}
    groups = [
        UniformTypeKVCacheSpecs.from_specs({name: padded[name] for name in mla_names}),
    ]
    if state_specs:
        groups.append(UniformTypeKVCacheSpecs.from_specs(state_specs))

    # Transpose the physical tuples. Each scheduler SWA group takes one layer
    # from every tuple, so its members use distinct slots at the same block ID.
    swa_columns = [
        [
            name
            for name in layer_tuple
            if isinstance(padded[name], AscendSlidingWindowMLASpec) and ".mtp." not in f".{name}"
        ]
        for layer_tuple in layer_tuples
    ]
    for column in zip_longest(*swa_columns):
        names = [name for name in column if name is not None]
        groups.append(UniformTypeKVCacheSpecs.from_specs({name: padded[name] for name in names}))

    draft_names = [
        name
        for layer_tuple in layer_tuples
        for name in layer_tuple
        if isinstance(padded[name], AscendSlidingWindowMLASpec) and ".mtp." in f".{name}"
    ]
    if draft_names:
        groups.append(UniformTypeKVCacheSpecs.from_specs({name: padded[name] for name in draft_names}))
    return groups


def make_cache_groups(grouped_specs, *, vllm_config: VllmConfig | None = None):
    """Build scheduler groups; ring is only transferable in a PD deployment.

    ``vllm_config`` 只用来判定 PD（见 ``ring_transfer_enabled``）。省略时
    等价于非 PD：ring group 仍是 ``enable_kv_transfer=False``，与 T-RING
    的改动前逐字节一致（UT 用单参数调用也走这条）。
    """
    groups = []
    for grouped in grouped_specs:
        members = tuple(grouped.kv_cache_specs.values())
        # ring 是每请求 scratch（prefix_cacheable=False）：非 PD 部署把它的
        # group 标成不可外部传输，免得 connector 去搬每请求私有的 ring 页。
        # PD/CED 相反——P 的 compressor state 必须交给 D（见 ring_transfer_enabled）。
        is_ring_group = bool(members) and all(is_ring_state_spec(spec) for spec in members)
        transferable = ring_transfer_enabled(vllm_config) if is_ring_group else True
        if is_ring_group and transferable:
            logger.info(
                "DeepSeek V4.1 ring group participates in KV transfer "
                "(PD deployment): %d compressor-state layers",
                len(members),
            )
        groups.append(
            KVCacheGroupSpec(
                layer_names=list(grouped.kv_cache_specs),
                kv_cache_spec=grouped,
                enable_kv_transfer=transferable,
            )
        )
    return groups


def _specs_from_groups(groups):
    specs = {}
    for group in groups:
        for name in group.layer_names:
            group_spec = group.kv_cache_spec
            specs[name] = (
                group_spec.kv_cache_specs[name] if isinstance(group_spec, UniformTypeKVCacheSpecs) else group_spec
            )
    return specs


def get_deepseek_v41_pool_bytes_per_block(groups):
    return sum(slot.page_size for slot in plan_cache_slots(_specs_from_groups(groups)))


def get_deepseek_v41_cache_offsets(groups):
    specs = _specs_from_groups(groups)
    offsets = {
        placement.name: placement.offset
        for slot in plan_cache_slots(specs)
        for placement in slot.placements
    }
    # 摘出池的 ring 不占 slot，它的 view 从自己那份 tensor 的 0 偏移开始。
    offsets.update({name: 0 for name, spec in specs.items() if is_ring_state_spec(spec)})
    return offsets


def is_ring_cache_group(group) -> bool:
    """该 KV cache group 是否就是「摘出池」的 ring（决定用哪个 BlockPool）。"""
    group_spec = group.kv_cache_spec
    specs = group_spec.kv_cache_specs.values() if isinstance(group_spec, UniformTypeKVCacheSpecs) else (group_spec,)
    specs = tuple(specs)
    return bool(specs) and all(is_ring_state_spec(spec) for spec in specs)


def get_deepseek_v41_ring_num_blocks(kv_cache_config) -> int:
    """ring 独立池的页数；0 表示这份配置里没有摘出池的 ring。

    页数从 ring 自己的 KVCacheTensor 描述符反推（size // block_stride），
    与 worker 侧 reshape 用的是同一个来源。
    """
    specs = _specs_from_groups(kv_cache_config.kv_cache_groups)
    for descriptor in kv_cache_config.kv_cache_tensors:
        if len(descriptor.layers) != 1:
            continue
        spec = specs.get(descriptor.layers[0])
        if spec is None or not is_ring_state_spec(spec):
            continue
        stride = int(descriptor.block_stride)
        if stride <= 0 or descriptor.size % stride:
            raise ValueError("out-of-pool ring tensor size is not a whole number of pages")
        return descriptor.size // stride
    return 0


def get_deepseek_v41_kv_cache_config(
    vllm_config: VllmConfig,
    groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """Allocate independent layer slots backed by one global block-ID pool."""
    specs = _specs_from_groups(groups)
    slots = plan_cache_slots(specs)
    if not slots or any(slot.page_size <= 0 for slot in slots):
        raise ValueError("DeepSeek V4.1 requires non-empty positive cache slots")
    ring_names = [name for name, spec in specs.items() if is_ring_state_spec(spec)]
    ring_tensors: list[KVCacheTensor] = []
    ring_bytes = 0
    if ring_names:
        ring_pages = {specs[name].page_size_bytes for name in ring_names}
        if len(ring_pages) != 1:
            raise ValueError("out-of-pool compressor state rings must share one page size")
        ring_page = ring_pages.pop()
        ring_num_blocks = ring_pool_num_blocks(vllm_config)
        ring_bytes = ring_num_blocks * ring_page * len(ring_names)
        # 每个 ring 层一份独立 tensor，页数 = 并发请求数 + 1（null）。
        # worker 侧 reshape 用 size // block_stride 还原页数
        # （model_runner.py 的 layer_name in layer_tuple_strides 分支）。
        ring_tensors = [
            KVCacheTensor(
                size=ring_num_blocks * ring_page,
                layers=[name],
                offset=0,
                layer_stride=0,
                block_stride=ring_page,
            )
            for name in ring_names
        ]
        logger.info(
            "DeepSeek V4.1 ring out of pool: %d pages x %d B x %d layers = %.2f MiB (max_num_seqs=%d)",
            ring_num_blocks,
            ring_page,
            len(ring_names),
            ring_bytes / (1 << 20),
            ring_num_blocks - 1,
        )
    pool_bytes = sum(slot.page_size for slot in slots)
    if ring_tensors and not is_a3_soft_kv8():
        # gate 与 SOFT_KV8 正交：bf16 时 SWA 页 131,072 仍然顶住地板，ring
        # 摘出去不会降 slot 页，只是白拿 48 MiB 的独立 backing。
        logger.warning(
            "VLLM_ASCEND_RING_OUT_OF_POOL=1 without VLLM_ASCEND_A3_SOFT_KV8=1: "
            "slot floor stays at %d B/block, so the %.2f MiB ring backing is a net loss",
            pool_bytes,
            ring_bytes / (1 << 20),
        )
    # ring 的独立 tensor 与每 block 的池子同预算：先扣掉它再算块数。
    capacity = max((available_memory - ring_bytes) // pool_bytes, 0)
    # Some device kernels form byte offsets in 32 bits. Bound the end of
    # every page, not just its starting address, and never pin a memory size.
    safe_blocks = min(KV_ADDRESS_SPACE_BYTES // slot.page_size for slot in slots)
    if ring_tensors:
        safe_blocks = min(safe_blocks, KV_ADDRESS_SPACE_BYTES // ring_tensors[0].block_stride)
    capacity = min(capacity, safe_blocks)
    num_blocks = may_override_num_blocks(vllm_config, capacity)
    if num_blocks < 2 or num_blocks > capacity:
        raise ValueError("DeepSeek V4.1 block override exceeds available memory or the 32-bit address limit")
    logger.info(
        "DeepSeek V4.1 KV slots: strides=%s, safe blocks=%d, allocated blocks=%d",
        [s.page_size for s in slots],
        safe_blocks,
        num_blocks,
    )
    if ring_tensors:
        logger.info(
            "DeepSeek V4.1 ring tensors: %s",
            [(t.layers[0].rsplit(".layers.", 1)[1].split(".", 1)[0], t.size) for t in ring_tensors],
        )
    tensors: list[KVCacheTensor] = []
    for slot in slots:
        page_size = slot.page_size
        layer_names = [p.name for p in slot.placements]
        size = num_blocks * page_size
        tensors.append(
            KVCacheTensor(
                size=size,
                layers=list(layer_names),
                offset=0,
                layer_stride=0,
                block_stride=page_size,
            )
        )
    tensors.extend(ring_tensors)
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
    )
