"""Unit tests for asynchronous prefill lookahead barriers."""

import importlib.util
import pathlib
import types
import unittest

SOURCE = pathlib.Path(__file__).resolve().parents[4] / "vllm_ascend/patch/worker/v41_prefill_policy.py"
spec = importlib.util.spec_from_file_location("v41_prefill_policy", SOURCE)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class PrefillPolicyTest(unittest.TestCase):
    def test_only_unresolved_prefill_limits_async_lookahead(self):
        prefill = types.SimpleNamespace(num_computed_tokens=10, num_prompt_tokens=20)
        decode = types.SimpleNamespace(num_computed_tokens=30, num_prompt_tokens=20)
        ids = policy.prefill_ids({"p": prefill, "d": decode}, True)
        self.assertEqual(ids, {"p"})
        output = types.SimpleNamespace(num_scheduled_tokens={"p": 10, "d": 1})
        self.assertTrue(policy.tag_output(output, ids))
        self.assertTrue(policy.pending_prefill([(None, output)], True))
        self.assertFalse(policy.pending_prefill([(None, output)], False))
        self.assertFalse(policy.pending_prefill([], True))
        decoded = types.SimpleNamespace(num_scheduled_tokens={"d": 1})
        self.assertFalse(policy.tag_output(decoded, ids))
        self.assertFalse(policy.pending_prefill([(None, decoded)], True))
        self.assertEqual(policy.prefill_ids({"p": prefill}, False), set())
