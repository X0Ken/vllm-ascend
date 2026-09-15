# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec
from vllm_ascend.core.single_type_kv_cache_manager import CompressAttentionManager
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import AscendHybridKVCacheCoordinator

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("use_eagle", [False, True])
@pytest.mark.parametrize("swa_hit_tokens", [0, 16384, 65536])
@pytest.mark.parametrize("reverse_full_groups", [False, True])
def test_full_attention_hits_match_common_length(use_eagle, swa_hit_tokens, reverse_full_groups):
    block_size = 128
    pool = BlockPool(4096, enable_caching=True, hash_block_size=block_size)
    specs = [
        AscendMLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            compress_ratio=ratio,
            model_version="deepseek_v4",
        )
        for ratio in (4, 128)
    ]
    if reverse_full_groups:
        specs.reverse()
    specs.append(
        AscendSlidingWindowMLASpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            sliding_window=256,
            model_version="deepseek_v4",
        )
    )
    hashes = [i.to_bytes(16, "big") for i in range(768)]
    managers = []
    attention_groups = []
    for gid, spec in enumerate(specs):
        manager_cls = CompressAttentionManager if gid < 2 else SlidingWindowManager
        managers.append(
            manager_cls(
                spec,
                block_pool=pool,
                enable_caching=True,
                kv_cache_group_id=gid,
                scheduler_block_size=16384,
            )
        )
        attention_groups.append((spec, [gid], manager_cls))
        span = block_size * (spec.compress_ratio if gid < 2 else 1)
        cached_tokens = 81920 if gid < 2 else swa_hit_tokens + use_eagle * block_size
        blocks = pool.get_new_blocks(cached_tokens // span)
        for i, block in enumerate(blocks):
            key = make_block_hash_with_group_id(hashes[(i + 1) * (span // block_size) - 1], gid)
            block.set_block_hash(key)
            pool.cached_block_hash_to_block.insert(key, block)
        pool.free_blocks(blocks)

    # Supply deterministic cache contents without model loading or NPU tensors.
    coordinator = AscendHybridKVCacheCoordinator.__new__(AscendHybridKVCacheCoordinator)
    coordinator.kv_cache_config = KVCacheConfig(
        num_blocks=4096,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([str(gid)], spec) for gid, spec in enumerate(specs)],
    )
    coordinator.block_pool = pool
    coordinator.hash_block_size = block_size
    coordinator.dcp_world_size = 1
    coordinator.enable_caching = True
    coordinator.enable_partial_hash_hits = False
    coordinator.lcm_block_size = 16384
    coordinator.scheduler_block_size = 16384
    coordinator.attention_groups = attention_groups
    coordinator.eagle_attn_group_indices = {0, 1, 2} if use_eagle else set()
    coordinator.single_type_managers = tuple(managers)

    hits, hit_length = coordinator.find_longest_cache_hit(hashes, 98304)
    assert hit_length == swa_hit_tokens
    for gid, spec in enumerate(specs[:2]):
        assert len(hits[gid]) * spec.block_size * spec.compress_ratio == hit_length

    coordinator.allocate_new_computed_blocks("external-hit", hits, hit_length, 16384)
    assert pool.get_num_free_blocks() == len(pool.free_block_queue.get_all_free_blocks())
    remaining = pool.get_new_blocks(pool.get_num_free_blocks())
    assert all(not block.is_null for block in remaining)
    pool.free_blocks(remaining)
    for manager in managers:
        manager.free("external-hit")
    assert pool.get_num_free_blocks() == 4095
    assert len(pool.free_block_queue.get_all_free_blocks()) == 4095
