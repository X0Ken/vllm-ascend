# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Framework-side V4.1 layer-outermost cache placement and allocation."""

from dataclasses import dataclass, replace

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
    """Select the V4.1 cache ABI from the detected hardware profile."""
    return get_current_hardware_profile().supports(HardwareCapability.DSV41_PACKED_CACHE)


def make_swa_cache_spec(*, block_size, window_size, head_size, dtype, cache_dtype, bounded_replay=False):
    if uses_a5_packed_cache():
        if head_size != MLA_HEAD_DIM:
            raise ValueError(f"A5 DeepSeek V4.1 requires head size {MLA_HEAD_DIM}, got {head_size}")
        # FP8 values followed by one BF16 scale per 32 values.
        head_size += (head_size // 32) * torch.bfloat16.itemsize
        dtype = torch.uint8
        cache_dtype = "a5_mxfp8_bf16_scale"
    return AscendSlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        sliding_window=window_size,
        cache_dtype_str=cache_dtype,
        model_version="deepseek_v41",
        alignment=None,
        **declared_kwarg(AscendSlidingWindowMLASpec, "bounded_replay", bounded_replay),
    )


def make_mla_cache_spec(*, block_size, head_size, compress_ratio):
    if uses_a5_packed_cache():
        if head_size != MLA_HEAD_DIM:
            raise ValueError(f"A5 DeepSeek V4.1 requires head size {MLA_HEAD_DIM}, got {head_size}")
        # Two FP4 values per byte, followed by one BF16 scale per 16 values.
        head_size = head_size // 2 + (head_size // 16) * torch.bfloat16.itemsize
        dtype = torch.uint8
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
    if uses_a5_packed_cache():
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
    if len(names) != len(set(names)) or set(names) != set(specs):
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
    groups = [
        UniformTypeKVCacheSpecs.from_specs({name: padded[name] for name in mla_names}),
        UniformTypeKVCacheSpecs.from_specs({name: padded[name] for name in state_names}),
    ]

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
    for names in zip(*swa_columns):
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


def make_cache_groups(grouped_specs):
    return [KVCacheGroupSpec(layer_names=list(s.kv_cache_specs), kv_cache_spec=s) for s in grouped_specs]


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
    return {
        placement.name: placement.offset
        for slot in plan_cache_slots(_specs_from_groups(groups))
        for placement in slot.placements
    }


def get_deepseek_v41_kv_cache_config(
    vllm_config: VllmConfig,
    groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """Allocate independent layer slots backed by one global block-ID pool."""
    slots = plan_cache_slots(_specs_from_groups(groups))
    if not slots or any(slot.page_size <= 0 for slot in slots):
        raise ValueError("DeepSeek V4.1 requires non-empty positive cache slots")
    capacity = max(available_memory // sum(slot.page_size for slot in slots), 0)
    # Some device kernels form byte offsets in 32 bits. Bound the end of
    # every page, not just its starting address, and never pin a memory size.
    safe_blocks = min(KV_ADDRESS_SPACE_BYTES // slot.page_size for slot in slots)
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
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
    )
