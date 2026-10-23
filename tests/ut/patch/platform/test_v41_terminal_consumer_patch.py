# SPDX-License-Identifier: Apache-2.0
"""Exercise real upstream admission guards without an NPU or model weights."""

import ast
import importlib.util
import os
import sys
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch


@unittest.skipUnless(os.getenv("VLLM_SOURCE_ROOT"), "requires patched pinned upstream source")
class TestTerminalConsumerAdmission(unittest.TestCase):
    def run_case(self, **changes):
        source = (Path(os.environ["VLLM_SOURCE_ROOT"]) / "vllm/v1/worker/gpu/model_runner.py").read_text()
        start = source.index("        parallel = self.vllm_config.parallel_config\n", source.index("model_inputs"))
        end = source.index("        # Update the EPLB meta.", start)
        snippet = textwrap.dedent(source[start:end])
        ast.parse(snippet)
        fields = dict(
            tp=8,
            dp=1,
            pp=1,
            pcp=1,
            dcp=1,
            sp=False,
            aux=True,
            scoring=False,
            prefill=True,
            all_prefill=True,
            graph_none=True,
            dummy=False,
            profile=False,
            tokens=512,
            outputs=1,
            lora=False,
            ubatch=False,
            ep=True,
            prefix_cache=False,
            connector=False,
        )
        fields.update(changes)
        parallel = NS(
            tensor_parallel_size=fields["tp"],
            data_parallel_size=fields["dp"],
            pipeline_parallel_size=fields["pp"],
            prefill_context_parallel_size=fields["pcp"],
            decode_context_parallel_size=fields["dcp"],
            use_sequence_parallel_moe=fields["sp"],
            enable_expert_parallel=fields["ep"],
        )
        worker = NS(
            model=NS(language_model=NS(supports_consumer_ffn_compaction=True)),
            vllm_config=NS(
                parallel_config=parallel,
                cache_config=NS(enable_prefix_caching=fields["prefix_cache"]),
                kv_transfer_config=object() if fields["connector"] else None,
            ),
            model_config=NS(hf_text_config=NS(sliding_window=128)),
            use_aux_hidden_state_outputs=fields["aux"],
            lora_config=object() if fields["lora"] else None,
            prompt_logprobs_worker=NS(uses_prompt_logprobs=NS(any=lambda: fields["scoring"])),
            _consumer_inputs_logged=True,
            device="npu",
        )
        batch = NS(
            has_prefill=fields["prefill"],
            is_prefilling_np=[fields["all_prefill"]],
            num_tokens=fields["tokens"],
            logits_indices=NS(numel=lambda: fields["outputs"]),
            num_reqs=2,
        )

        class QueryStarts(list):
            def tolist(self):
                return list(self)

            def __getitem__(self, item):
                value = super().__getitem__(item)
                return type(self)(value) if isinstance(item, slice) else value

        batch.query_start_loc_np = QueryStarts([0, fields["tokens"] // 2, fields["tokens"]])
        scope = dict(
            self=worker,
            input_batch=batch,
            dummy_run=fields["dummy"],
            is_profile=fields["profile"],
            batch_desc=NS(cg_mode=0 if fields["graph_none"] else 1),
            CUDAGraphMode=NS(NONE=0),
            np=NS(all=all),
            model_inputs={},
            skip_compiled=False,
            ubatch_state=object() if fields["ubatch"] else None,
            torch=NS(tensor=lambda value, **kwargs: value, int64="int64", int32="int32"),
        )
        path = Path(__file__).resolve().parents[4] / "vllm_ascend/models/deepseek_v41/consumer_compaction.py"
        spec = importlib.util.spec_from_file_location("terminal_consumer_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {spec.name: module, "vllm_ascend.models.deepseek_v41.consumer_compaction": module},
        ):
            spec.loader.exec_module(module)
            exec(snippet, scope)
        return scope

    def test_eligible_prefill_selects_raw_forward_and_output_rows(self):
        result = self.run_case()
        self.assertIn("consumer_output_indices", result["model_inputs"])
        self.assertTrue(result["skip_compiled"])

    def test_decode_capture_scoring_and_parallel_fallbacks_keep_original_path(self):
        cases = [
            dict(prefill=False),
            dict(all_prefill=False),
            dict(graph_none=False),
            dict(dummy=True),
            dict(profile=True),
            dict(aux=False),
            dict(scoring=True),
            dict(tp=4),
            dict(dp=2),
            dict(pp=2),
            dict(pcp=2),
            dict(dcp=2),
            dict(sp=True),
            dict(tokens=256),
            dict(outputs=256),
            dict(lora=True),
            dict(ubatch=True),
            dict(ep=False),
        ]
        for changes in cases:
            with self.subTest(changes=changes):
                result = self.run_case(**changes)
                self.assertEqual(result["model_inputs"], {})
                self.assertFalse(result["skip_compiled"])

    def test_prefix_and_connector_reuse_keep_all_attention_cache_writes(self):
        for changes in (dict(prefix_cache=True), dict(connector=True), dict(prefix_cache=True, connector=True)):
            with self.subTest(changes=changes):
                result = self.run_case(outputs=2, **changes)
                # FFN compaction is row-local and does not write cache pages.
                self.assertIn("consumer_output_indices", result["model_inputs"])
                self.assertNotIn("consumer_tail_indices", result["model_inputs"])
                self.assertTrue(result["skip_compiled"])

    def test_cache_free_prefill_can_still_trim_terminal_attention(self):
        result = self.run_case(outputs=2)
        self.assertIn("consumer_tail_indices", result["model_inputs"])
        self.assertEqual(len(result["model_inputs"]["consumer_tail_indices"]), 256)
