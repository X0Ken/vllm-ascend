# SPDX-License-Identifier: Apache-2.0
"""Exercise real upstream admission guards without an NPU or model weights."""

import ast
import os
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace as NS


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
            vllm_config=NS(parallel_config=parallel),
            use_aux_hidden_state_outputs=fields["aux"],
            lora_config=object() if fields["lora"] else None,
            prompt_logprobs_worker=NS(uses_prompt_logprobs=NS(any=lambda: fields["scoring"])),
            _consumer_inputs_logged=True,
        )
        batch = NS(
            has_prefill=fields["prefill"],
            is_prefilling_np=[fields["all_prefill"]],
            num_tokens=fields["tokens"],
            logits_indices=NS(numel=lambda: fields["outputs"]),
            num_reqs=2,
        )
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
        )
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
