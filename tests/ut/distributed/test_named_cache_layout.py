# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DMA planning invariants; these tests require neither torch nor an NPU."""

import importlib.util
import pathlib
import sys
import unittest
from dataclasses import replace

PATH = pathlib.Path(__file__).resolve().parents[3] / "vllm_ascend/distributed/kv_transfer/kv_p2p/named_cache_layout.py"
SPEC = importlib.util.spec_from_file_location("named_cache_layout_test_module", PATH)
layout = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = layout
SPEC.loader.exec_module(layout)


class FakeTensor:
    shape = (32, 4, 8)
    dtype = "bfloat16"

    def stride(self):
        return (128, 16, 1)

    def element_size(self):
        return 2

    def data_ptr(self):
        return 1 << 44


class NamedTransferTests(unittest.TestCase):
    def setUp(self):
        self.local = layout.describe_plane("layers.20.long_kv", 0, 1, FakeTensor(), 32)
        self.remote = replace(self.local, group=0, base=1 << 45, stride=512)

    def test_strided_views_copy_logical_segments_only(self):
        self.assertEqual(self.local.segments, ((0, 16), (32, 16), (64, 16), (96, 16)))
        destinations, sources, lengths = layout.plan_named_transfer(
            [self.local],
            [self.remote],
            [[], [3]],
            [[7]],
        )
        self.assertEqual(destinations, [(1 << 44) + 3 * 256 + offset for offset in (0, 32, 64, 96)])
        self.assertEqual(sources, [(1 << 45) + 7 * 512 + offset for offset in (0, 32, 64, 96)])
        self.assertEqual(lengths, [16] * 4)

    def test_semantic_mapping_survives_group_reordering(self):
        extra = replace(self.remote, name="other", group=1)
        got = layout.plan_named_transfer([self.local], [extra, self.remote], [[], [2]], [[5], [9]])
        self.assertEqual(got[1][0], self.remote.base + 5 * self.remote.stride)

    def test_offsets_above_four_gib_do_not_wrap(self):
        dst = replace(self.local, blocks=40000, stride=147712)
        src = replace(self.remote, blocks=40000, stride=147712)
        got = layout.plan_named_transfer([dst], [src], [[], [35000]], [[35000]])
        self.assertGreater(got[0][0] - dst.base, 1 << 32)

    def test_missing_required_plane_fails(self):
        with self.assertRaisesRegex(ValueError, "Missing remote"):
            layout.plan_named_transfer([self.local], [], [[], [2]], [[3]])

    def test_missing_optional_plane_is_explicit(self):
        self.assertEqual(
            layout.plan_named_transfer(
                [self.local],
                [],
                [[], [2]],
                [[3]],
                optional_names={self.local.name},
            ),
            ([], [], []),
        )

    def test_dtype_and_shape_mismatch_fail(self):
        for remote in [replace(self.remote, dtype="int8"), replace(self.remote, shape=(4, 9))]:
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                layout.plan_named_transfer([self.local], [remote], [[], [2]], [[3]])

    def test_null_negative_and_out_of_bounds_ids_fail(self):
        for block in [0, -1, 32]:
            with self.assertRaisesRegex(ValueError, "block ID"):
                layout.plan_named_transfer([self.local], [self.remote], [[], [block]], [[3]])

    def test_duplicate_and_insufficient_remote_blocks_fail(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            layout.plan_named_transfer([self.local], [self.remote, self.remote], [[], [2]], [[3]])
        with self.assertRaisesRegex(ValueError, "too short"):
            layout.plan_named_transfer([self.local], [self.remote], [[], [2, 3]], [[3]])

    def test_block_aligned_receive_does_not_shift_the_whole_prefix(self):
        # P has the partial final block; D loads only complete blocks. Copy
        # P[0:3], never P[-3:], so the beginning of the prompt survives.
        got = layout.plan_named_transfer(
            [self.local],
            [self.remote],
            [[], [1, 2, 3]],
            [[5, 6, 7, 8]],
            local_offsets=[0, 0],
        )
        self.assertEqual(got[1][0], self.remote.base + 5 * self.remote.stride)
        self.assertEqual(got[1][8], self.remote.base + 7 * self.remote.stride)

    def test_local_prefix_hit_uses_logical_offset_in_remote_block_list(self):
        got = layout.plan_named_transfer(
            [self.local],
            [self.remote],
            [[], [1, 2]],
            [[5, 6, 7, 8]],
            local_offsets=[0, 1],
        )
        self.assertEqual(got[1][0], self.remote.base + 6 * self.remote.stride)
        self.assertEqual(got[1][4], self.remote.base + 7 * self.remote.stride)

    def test_wire_roundtrip_preserves_layout(self):
        self.assertEqual(layout.CachePlane.from_dict(self.local.to_dict()), self.local)

    def test_cross_block_plane_fails(self):
        tensor = FakeTensor()
        tensor.stride = lambda: (8, 16, 1)
        with self.assertRaisesRegex(ValueError, "crosses"):
            layout.describe_plane("bad", 0, 0, tensor, 32)


if __name__ == "__main__":
    unittest.main()
