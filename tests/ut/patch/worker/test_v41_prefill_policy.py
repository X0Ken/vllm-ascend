"""Unit tests for asynchronous prefill lookahead barriers."""

import importlib.util
import pathlib
import types
import unittest
from unittest.mock import Mock

from vllm_ascend.config_utils import dsv41_admission_config

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

    def test_first_decode_uses_validated_real_drafts_after_prefill_drains(self):
        output = types.SimpleNamespace(
            has_structured_output_requests=True,
            pending_structured_output_tokens=False,
            scheduled_spec_decode_tokens={"request": [-1] * 5},
        )
        drafts = types.SimpleNamespace(req_ids=["request"], draft_token_ids=[[30, 128825, 10699, 1018, 30]])
        engine = types.SimpleNamespace(
            async_scheduling=True,
            check_for_draft_tokens=True,
            model_executor=Mock(),
            scheduler=Mock(),
        )
        engine.model_executor.take_draft_token_ids.return_value = drafts

        def validate_and_update(real_drafts, scheduled_output):
            scheduled_output.scheduled_spec_decode_tokens = dict(zip(real_drafts.req_ids, real_drafts.draft_token_ids))

        engine.scheduler.update_draft_token_ids_in_output.side_effect = validate_and_update
        policy.backfill_structured_drafts(engine, output)
        self.assertEqual(output.scheduled_spec_decode_tokens["request"], drafts.draft_token_ids[0])
        engine.scheduler.update_draft_token_ids_in_output.assert_called_once_with(drafts, output)

    def test_plain_prefill_and_synchronous_batches_do_not_fetch_drafts(self):
        for asynchronous, speculative, structured, tokens in [
            (False, True, True, {"request": [-1] * 5}),
            (True, False, True, {"request": [-1] * 5}),
            (True, True, False, {"request": [-1] * 5}),
            (True, True, True, {}),
        ]:
            with self.subTest(
                async_scheduling=asynchronous,
                speculative=speculative,
                structured=structured,
                tokens=tokens,
            ):
                engine = types.SimpleNamespace(
                    async_scheduling=asynchronous,
                    check_for_draft_tokens=speculative,
                    model_executor=Mock(),
                    scheduler=Mock(),
                )
                output = types.SimpleNamespace(
                    has_structured_output_requests=structured,
                    scheduled_spec_decode_tokens=tokens,
                )
                policy.backfill_structured_drafts(engine, output)
                engine.model_executor.take_draft_token_ids.assert_not_called()
                engine.scheduler.update_draft_token_ids_in_output.assert_not_called()


class AdmissionConfigTest(unittest.TestCase):
    def test_disabled_does_not_require_a_worker_singleton(self):
        self.assertEqual(dsv41_admission_config(types.SimpleNamespace(additional_config={})), (False, 1))

    def test_enabled_and_positive_batch_limit(self):
        runtime = types.SimpleNamespace(
            additional_config={"enable_dsv41_admission_gate": True, "dsv41_gate_max_prefills": 8}
        )
        self.assertEqual(dsv41_admission_config(runtime), (True, 8))

    def test_invalid_batch_limits_are_rejected(self):
        for maximum in (0, -1, True, 1.5, "8"):
            runtime = types.SimpleNamespace(
                additional_config={"enable_dsv41_admission_gate": True, "dsv41_gate_max_prefills": maximum}
            )
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                dsv41_admission_config(runtime)
