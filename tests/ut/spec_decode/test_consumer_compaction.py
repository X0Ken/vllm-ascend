# SPDX-License-Identifier: Apache-2.0
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

source = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/consumer_compaction.py"
spec = importlib.util.spec_from_file_location("consumer_compaction", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
CONFIG = dict(
    num_hidden_layers=40,
    sliding_window=128,
    kv_source_layer_ids=[2, 8, 14, 20],
    index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
    engram_layer_ids=[1, 14],
    dspark_target_layer_ids=[37, 38, 39],
    candidate_source_layer_id=20,
)


class ConsumerCompactionTests(unittest.TestCase):
    def test_only_unpacked_terminal_c1_consumer_has_capability(self):
        role = dict(compress_ratio=1, is_kv_source=False, is_index_source=False)
        attention = SimpleNamespace(role=SimpleNamespace(**role), packed_cache_ops=None)
        self.assertTrue(module.supports_terminal_consumer(attention))
        self.assertFalse(module.supports_terminal_consumer(None))
        attention.packed_cache_ops = object()
        self.assertFalse(module.supports_terminal_consumer(attention))
        for field, value in [("compress_ratio", 2), ("is_kv_source", True), ("is_index_source", True)]:
            changed = dict(role, **{field: value})
            attention = SimpleNamespace(role=SimpleNamespace(**changed), packed_cache_ops=None)
            self.assertFalse(module.supports_terminal_consumer(attention))

    def test_real_topology_retains_late_aux(self):
        p = module.plan_consumer_compaction(CONFIG, 8192, requires_full_hc=False, compact_metadata_supported=True)
        self.assertEqual((p.boundary, p.remaining_layers, p.retained_tokens), (39, 1, 128))
        self.assertTrue(p.eligible)

    def test_metadata_and_hc_contracts_block_activation(self):
        a = module.plan_consumer_compaction(CONFIG, 8192, requires_full_hc=True, compact_metadata_supported=True)
        b = module.plan_consumer_compaction(CONFIG, 8192, requires_full_hc=False, compact_metadata_supported=False)
        self.assertFalse(a.eligible)
        self.assertFalse(b.eligible)
        self.assertIn("Full-HC", a.reason)
        self.assertIn("metadata", b.reason)

    def test_multiple_local_layers_need_expanded_history(self):
        config = dict(CONFIG, dspark_target_layer_ids=[], index_source_layer_ids=[20])
        p = module.plan_consumer_compaction(config, 100000, requires_full_hc=False, compact_metadata_supported=True)
        self.assertEqual(p.retained_tokens, 1 + 19 * 127)
        self.assertFalse(
            module.plan_consumer_compaction(
                config, 100, requires_full_hc=False, compact_metadata_supported=True
            ).eligible
        )


class TerminalSuffixTests(unittest.TestCase):
    def test_mixed_prompt_lengths_preserve_absolute_rows(self):
        rows, starts, outputs = module.terminal_suffix_layout([0, 512, 1536], 128)
        self.assertEqual(rows, list(range(384, 512)) + list(range(1408, 1536)))
        self.assertEqual(starts, [0, 128, 256])
        self.assertEqual(outputs, [127, 255])
        self.assertEqual([rows[index] for index in outputs], [511, 1535])

    def test_short_or_untrimmed_chunks_fall_back(self):
        self.assertIsNone(module.terminal_suffix_layout([0, 64, 1024], 128))
        self.assertIsNone(module.terminal_suffix_layout([0, 128, 256], 128))
        with self.assertRaises(ValueError):
            module.terminal_suffix_layout([0, 128, 128], 128)


if __name__ == "__main__":
    unittest.main()
