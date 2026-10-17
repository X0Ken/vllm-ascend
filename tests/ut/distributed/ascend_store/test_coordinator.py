#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import inspect
import unittest
from dataclasses import dataclass, replace
from unittest.mock import patch
from weakref import ref

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, MambaSpec, SlidingWindowSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import get_block_hashes
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import (
    AscendStoreCoordinator,
    ExternalCachedBlockPool,
    _reachable_block_mask,
)
# isort: on


def _hashes(num_blocks: int) -> list[bytes]:
    return [bytes([idx % 251]) * 32 for idx in range(num_blocks)]


def _full_spec(block_size: int) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )


def _sliding_spec(block_size: int, sliding_window: int) -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=sliding_window,
    )


@dataclass(frozen=True)
class _FakeCompressedSpec:
    block_size: int
    compress_ratio: int

    def copy_with_new_block_size(self, block_size):
        return replace(self, block_size=block_size)


class _FakeCompressedManager:
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes,
        max_length,
        kv_cache_group_ids,
        block_pool,
        kv_cache_spec,
        drop_eagle_block=False,
        alignment_tokens=16,
        **kwargs,
    ):
        computed: tuple[list[object], ...] = tuple([] for _ in kv_cache_group_ids)
        logical_block_size = kv_cache_spec.block_size
        if logical_block_size != block_pool.hash_block_size:
            scale_factor = logical_block_size // block_pool.hash_block_size
            block_hashes = [
                block_hashes[index + scale_factor - 1]
                for index in range(0, len(block_hashes) // scale_factor * scale_factor, scale_factor)
            ]
        max_blocks = max_length // logical_block_size
        for block_hash in list(block_hashes)[:max_blocks]:
            cached = block_pool.get_cached_block(block_hash, kv_cache_group_ids)
            if not cached:
                break
            for blocks, block in zip(computed, cached):
                blocks.append(block)
        return computed, len(computed[0]) * logical_block_size


class _FakePrefixManager:
    """FA manager: contiguous prefix walk over externally cached blocks."""

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes,
        max_length,
        kv_cache_group_ids,
        block_pool,
        kv_cache_spec,
        drop_eagle_block=False,
        alignment_tokens=16,
        **kwargs,
    ):
        computed: tuple[list[object], ...] = tuple([] for _ in kv_cache_group_ids)
        max_blocks = max_length // kv_cache_spec.block_size
        for block_hash in list(block_hashes)[:max_blocks]:
            cached = block_pool.get_cached_block(block_hash, kv_cache_group_ids)
            if not cached:
                break
            for blocks, block in zip(computed, cached):
                blocks.append(block)
        return computed, len(computed[0]) * kv_cache_spec.block_size


class TestAscendStoreCoordinator(unittest.TestCase):
    def test_compressed_group_hits_on_effective_granularity(self):
        block_hashes = _hashes(128)
        grouped_hash = get_block_hashes(block_hashes, group_block_size=128 * 128, hash_block_size=128)[0]
        coord = AscendStoreCoordinator(
            [KVCacheGroupSpec(["layer.0"], _full_spec(128 * 128))],
            scheduler_block_size=128 * 128,
            hash_block_size=128,
            group_block_sizes=[128 * 128],
            group_cache_families=["c128"],
        )

        _, hit_length = coord.find_longest_cache_hit(
            block_hashes,
            128 * 128,
            ExternalCachedBlockPool(128, {(0, bytes(grouped_hash))}),
        )

        self.assertEqual(hit_length, 128 * 128)

    def test_compressed_spec_does_not_apply_ratio_twice(self):
        block_hashes = _hashes(128)
        grouped_hash = get_block_hashes(block_hashes, group_block_size=128 * 128, hash_block_size=128)[0]

        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._get_manager_class",
            return_value=_FakeCompressedManager,
        ):
            coord = AscendStoreCoordinator(
                [
                    KVCacheGroupSpec(
                        ["layer.0"],
                        _FakeCompressedSpec(
                            block_size=128 * 128,
                            compress_ratio=128,
                        ),
                    )
                ],
                scheduler_block_size=128 * 128,
                hash_block_size=128,
                group_block_sizes=[128 * 128],
                group_cache_families=["c128"],
            )

            _, hit_length = coord.find_longest_cache_hit(
                block_hashes,
                128 * 128,
                ExternalCachedBlockPool(128, {(0, bytes(grouped_hash))}),
            )

        self.assertEqual(coord.group_effective_specs[0].compress_ratio, 128)
        self.assertEqual(hit_length, 128 * 128)

    def test_missing_required_group_returns_zero(self):
        block_hashes = _hashes(128)
        c1_exists = {(0, block_hash) for block_hash in block_hashes}
        coord = AscendStoreCoordinator(
            [
                KVCacheGroupSpec(["layer.0"], _full_spec(128)),
                KVCacheGroupSpec(["layer.1"], _full_spec(128 * 128)),
            ],
            scheduler_block_size=128 * 128,
            hash_block_size=128,
            group_block_sizes=[128, 128 * 128],
            group_cache_families=["c1", "c128"],
        )

        _, hit_length = coord.find_longest_cache_hit(
            block_hashes,
            128 * 128,
            ExternalCachedBlockPool(128, c1_exists),
        )

        self.assertEqual(hit_length, 0)
        for local, expected in ((16383, 0), (16384, 16384)):
            with self.subTest(local=local):
                self.assertEqual(
                    coord.find_reachable_hit_tokens(block_hashes, 16384, lambda *args: [], hbm_hit_tokens=local),
                    expected,
                )

    def test_store_mask_uses_manager_reachability(self):
        coord = AscendStoreCoordinator(
            [KVCacheGroupSpec(["layer.0"], _sliding_spec(block_size=128, sliding_window=256))],
            scheduler_block_size=512,
            hash_block_size=128,
            group_block_sizes=[128],
            group_cache_families=["c1"],
        )

        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._reachable_block_mask",
            return_value=[False, False, False, True],
        ):
            masks = coord.store_mask(512)

        self.assertEqual(masks, ([False, False, False, True],))

    def test_lookup_mask_uses_reachability_without_retention(self):
        coord = AscendStoreCoordinator(
            [KVCacheGroupSpec(["layer.0"], _sliding_spec(block_size=128, sliding_window=256))],
            scheduler_block_size=512,
            hash_block_size=128,
            group_block_sizes=[128],
            group_cache_families=["c1"],
            retention_interval=256,
        )
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._reachable_block_mask",
            return_value=[False, False, False, True],
        ) as reachable:
            masks = coord.lookup_mask(512)

        self.assertEqual(masks, ([False, False, False, True],))
        self.assertEqual(reachable.call_args.kwargs["retention_interval"], 256)

    def test_reachable_mask_preserves_optional_kwargs_for_flexible_signature(self):
        class FlexibleManager:
            @staticmethod
            def reachable_block_mask(
                start_block,
                end_block,
                alignment_tokens,
                kv_cache_spec,
                use_eagle,
                retention_interval=None,
                **kwargs,
            ):
                return [retention_interval, kwargs["num_prompt_tokens"]]

        result = _reachable_block_mask(
            FlexibleManager,
            start_block=0,
            end_block=8,
            alignment_tokens=128,
            kv_cache_spec=None,
            use_eagle=False,
            retention_interval=256,
            num_prompt_tokens=1024,
        )

        self.assertEqual(result, [256, 1024])

    def test_mamba_replay_boundary_and_partial_prefix_lookup(self):
        from vllm.v1.core.single_type_kv_cache_manager import MambaManager

        if "reachable_boundaries" not in inspect.signature(MambaManager.reachable_block_mask).parameters:
            self.skipTest("requires the reachable_boundaries vLLM API")
        spec = MambaSpec(block_size=256, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode="align")
        for interval, expected in [
            (0, [False, False, True, False]),
            (512, [False, True, True, True]),
            (None, [True, True, True, True]),
        ]:
            with self.subTest(retention_interval=interval):
                coord = AscendStoreCoordinator(
                    [KVCacheGroupSpec(["attention"], _full_spec(256)), KVCacheGroupSpec(["state"], spec)],
                    scheduler_block_size=256,
                    hash_block_size=256,
                    group_block_sizes=[256, 256],
                    group_cache_families=["default", "default"],
                    retention_interval=interval,
                )
                self.assertEqual(coord.store_mask(1024, num_prompt_tokens=769)[1], expected)
                hashes = _hashes(4)

                def query(group_id, group_hashes, mask):
                    # A shorter request published only its 768-token checkpoint.
                    return [
                        h for i, h in enumerate(group_hashes) if (mask is None or mask[i]) and (group_id == 0 or i == 2)
                    ]

                self.assertEqual(coord.find_reachable_hit_tokens(hashes, 1024, query), 768)

    def test_legacy_reachable_mask_ignores_new_boundary_argument(self):
        class LegacyManager:
            @staticmethod
            def reachable_block_mask(
                start_block,
                end_block,
                alignment_tokens,
                kv_cache_spec,
                use_eagle,
                retention_interval=None,
                num_prompt_tokens=None,
            ):
                return [retention_interval, num_prompt_tokens]

        result = _reachable_block_mask(
            LegacyManager,
            start_block=0,
            end_block=1,
            alignment_tokens=128,
            kv_cache_spec=None,
            use_eagle=False,
            retention_interval=0,
            num_prompt_tokens=129,
            reachable_boundaries=(128,),
        )
        self.assertEqual(result, [0, 129])

    def test_store_mask_propagates_eagle_to_same_spec_siblings(self):
        calls = []

        def fake_reachable_block_mask(*args, **kwargs):
            calls.append(kwargs["use_eagle"])
            return [True, False, True, False]

        shared_spec = _sliding_spec(block_size=128, sliding_window=256)
        coord = AscendStoreCoordinator(
            [
                KVCacheGroupSpec(["layer.0"], shared_spec),
                KVCacheGroupSpec(["layer.mtp"], shared_spec, is_eagle_group=True),
            ],
            scheduler_block_size=512,
            hash_block_size=128,
            group_block_sizes=[128, 128],
            group_cache_families=["c1", "c1"],
        )

        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._reachable_block_mask",
            side_effect=fake_reachable_block_mask,
        ):
            masks = coord.store_mask(512)

        self.assertEqual(calls, [True, True])
        self.assertEqual(masks, ([True, False, True, False], [True, False, True, False]))

    def test_reachable_mask_cache_preserves_inputs_and_caller_isolation(self):
        coord = AscendStoreCoordinator(
            [KVCacheGroupSpec(["layer.0"], _sliding_spec(16, 32))],
            scheduler_block_size=16,
            hash_block_size=16,
            group_block_sizes=[16],
            group_cache_families=["c1"],
        )
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._reachable_block_mask",
            side_effect=lambda *args, **kwargs: [True] + [False] * (kwargs["end_block"] - 1),
        ) as build_mask:
            coord.store_mask(32, 33)[0][0] = False
            self.assertEqual(coord.store_mask(32, 33), ([True, False],))
            coord.lookup_mask(32)[0][0] = False
            self.assertEqual(coord.lookup_mask(32), ([True, False],))
            self.assertEqual(build_mask.call_count, 2)
            coord.retention_interval = 64
            coord.store_mask(32, 33)
            coord.store_mask(48, 33)
            coord.store_mask(32, 65)
            self.assertEqual(build_mask.call_count, 5)

        cached_masks = coord._cached_reachable_masks
        coord_ref = ref(coord)
        del coord
        self.assertIsNone(coord_ref())
        self.assertGreater(cached_masks.cache_info().currsize, 0)

    def test_compressed_masks_use_full_attention_reachability(self):
        coord = AscendStoreCoordinator(
            [KVCacheGroupSpec(["layer.0"], _full_spec(block_size=512))],
            scheduler_block_size=2048,
            hash_block_size=128,
            group_block_sizes=[512],
            group_cache_families=["c4"],
        )

        self.assertEqual(
            coord.store_mask(2048, num_prompt_tokens=2048),
            ([True, True, True, True],),
        )
        self.assertEqual(coord.load_mask(_hashes(16), 2048), ([True] * 4,))

    def test_passes_mask_allowed_hashes_to_query_callback(self):
        coord = AscendStoreCoordinator(
            [KVCacheGroupSpec(["layer.0"], _sliding_spec(block_size=128, sliding_window=256))],
            scheduler_block_size=256,
            hash_block_size=128,
            group_block_sizes=[128],
            group_cache_families=["c1"],
        )
        calls: list[tuple[int, list, list | None]] = []

        def query_group_hits(group_id, group_block_hashes, lookup_mask):
            calls.append((group_id, list(group_block_hashes), list(lookup_mask) if lookup_mask is not None else None))
            return []

        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._reachable_block_mask",
            return_value=[False, True],
        ):
            hit = coord.find_reachable_hit_tokens(_hashes(2), 256, query_group_hits)

        self.assertEqual(hit, 0)
        self.assertEqual(calls, [(0, _hashes(2), [False, True])])

    def test_hit_length_derived_from_callback_hits(self):
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._get_manager_class",
            return_value=_FakePrefixManager,
        ):
            coord = AscendStoreCoordinator(
                [KVCacheGroupSpec(["layer.0"], _full_spec(128))],
                scheduler_block_size=256,
                hash_block_size=128,
                group_block_sizes=[128],
                group_cache_families=["c4"],
            )
        block_hashes = _hashes(4)
        received: list[list] = []

        def query_group_hits(group_id, group_block_hashes, lookup_mask):
            received.append(list(group_block_hashes))
            self.assertIsNone(lookup_mask)
            return list(group_block_hashes[:3])

        hit = coord.find_reachable_hit_tokens(block_hashes, 384, query_group_hits)

        # Only whole hash blocks within token_len are queried (384 // 128 = 3).
        self.assertEqual(received, [_hashes(3)])
        # Restore boundaries must be aligned even when the hash granularity is finer.
        self.assertEqual(hit, 256)

    def test_no_hits_returns_zero_without_deriving(self):
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._get_manager_class",
            return_value=_FakePrefixManager,
        ):
            coord = AscendStoreCoordinator(
                [KVCacheGroupSpec(["layer.0"], _full_spec(128))],
                scheduler_block_size=256,
                hash_block_size=128,
                group_block_sizes=[128],
                group_cache_families=["c4"],
            )

        with patch.object(coord, "find_longest_cache_hit") as derive:
            hit = coord.find_reachable_hit_tokens(_hashes(2), 256, lambda *args: [])

        self.assertEqual(hit, 0)
        derive.assert_not_called()

    def test_external_eagle_load_covers_distinct_target_and_draft_groups(self):
        spec = _sliding_spec(128, 128)
        coord = AscendStoreCoordinator(
            [
                KVCacheGroupSpec(["target"], spec),
                KVCacheGroupSpec(["draft"], replace(spec, head_size=2), is_eagle_group=True),
            ],
            scheduler_block_size=128,
            hash_block_size=128,
            group_block_sizes=[128, 128],
            group_cache_families=["swa", "swa"],
            use_eagle=True,
        )
        self.assertEqual(coord.load_mask(_hashes(4), 512), ([False, False, True, True], [False, False, True, True]))

    def test_external_eagle_missing_tail_falls_back_to_complete_checkpoint(self):
        coord = AscendStoreCoordinator(
            [KVCacheGroupSpec(["draft"], _sliding_spec(128, 128), is_eagle_group=True)],
            scheduler_block_size=128,
            hash_block_size=128,
            group_block_sizes=[128],
            group_cache_families=["swa"],
            use_eagle=True,
        )
        hashes = _hashes(6)
        # A complete older pair and an isolated latest block. The isolated
        # block cannot be restored as a reusable EAGLE checkpoint.
        existing = {hashes[1], hashes[2], hashes[5]}
        hit = coord.find_reachable_hit_tokens(hashes, 768, lambda group_id, group_hashes, mask: existing)
        self.assertEqual(hit, 384)

    def test_restore_intersects_sparse_windows_across_groups_and_ranks(self):
        spec = _sliding_spec(128, 128)
        coord = AscendStoreCoordinator(
            [
                KVCacheGroupSpec(["full"], _full_spec(128)),
                KVCacheGroupSpec(["target"], spec),
                KVCacheGroupSpec(["draft"], replace(spec, head_size=2), is_eagle_group=True),
            ],
            scheduler_block_size=128,
            hash_block_size=128,
            group_block_sizes=[128] * 3,
            group_cache_families=["c1", "swa", "swa"],
            use_eagle=True,
        )
        hashes = _hashes(6)
        a = {hashes[i] for i in [0, 1, 4, 5]}
        b = {hashes[i] for i in [0, 1, 3, 4]}
        hits = {0: [hashes, hashes], 1: [a, b], 2: [hashes, hashes]}
        # Per-rank maxima are 768 and 640. Their minimum, 640, is invalid
        # on the first rank. The latest common complete window ends at 256.
        self.assertEqual(coord.restorable_prefixes(hashes, 768, hits), [128, 256])
        for mask in coord.load_mask(hashes, 256):
            self.assertEqual(mask, [True, True])
        hits[2] = [set()]
        self.assertEqual(coord.restorable_prefixes(hashes, 768, hits), [])

    def test_restore_load_and_retention_share_coarse_alignment_window(self):
        spec = _sliding_spec(128, 128)
        coord = AscendStoreCoordinator(
            [
                KVCacheGroupSpec(["full"], _full_spec(256)),
                KVCacheGroupSpec(["target"], spec),
                KVCacheGroupSpec(["draft"], replace(spec, head_size=2), is_eagle_group=True),
            ],
            scheduler_block_size=256,
            hash_block_size=128,
            group_block_sizes=[256, 128, 128],
            group_cache_families=["c4", "default", "default"],
            use_eagle=True,
            retention_interval=0,
        )
        hashes = _hashes(8)
        tail = set(hashes[5:8])
        full_hashes = get_block_hashes(hashes, 256, 128)
        hits = {0: [full_hashes], 1: [tail], 2: [tail]}
        self.assertEqual(coord.restorable_prefixes(hashes, 1024, hits), [1024])
        masks = coord.load_mask(hashes, 1024)
        stored = coord.store_mask(1024, num_prompt_tokens=1025)
        self.assertEqual(masks[1], [False] * 5 + [True] * 3)
        for group in [1, 2]:
            self.assertTrue(all(not needed or kept for needed, kept in zip(masks[group], stored[group])))
        hits[1] = [tail - {hashes[5]}]
        self.assertEqual(coord.restorable_prefixes(hashes, 1024, hits), [])


if __name__ == "__main__":
    unittest.main()
