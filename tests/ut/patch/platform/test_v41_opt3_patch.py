# SPDX-License-Identifier: Apache-2.0
"""Run with VLLM_SOURCE_ROOT pointing at the patched, pinned vLLM source."""

import ast
import copy
import os
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace


@unittest.skipUnless(os.getenv("VLLM_SOURCE_ROOT"), "requires patched upstream source")
class TestOpt3CorePatch(unittest.TestCase):
    def setUp(self):
        root = Path(os.environ["VLLM_SOURCE_ROOT"])
        self.runner = (root / "vllm/v1/worker/gpu/model_runner.py").read_text()
        self.speculator = (root / "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py").read_text()

    def test_query_buckets_keep_target_and_model_objects_shared(self):
        start = self.speculator.index("        draft_config = self.vllm_config")
        end = self.speculator.index("        self.query_cudagraph_manager =", start)
        snippet = textwrap.dedent(self.speculator[start:end])
        for method, queries, sizes in [("dspark", 5, [2, 20]), ("dspark", 3, [3, 48]), ("dflash", 6, [])]:
            with self.subTest(method=method, queries=queries):
                model = object()
                target = SimpleNamespace(
                    compilation_config=SimpleNamespace(cudagraph_capture_sizes=list(sizes)),
                    speculative_config=SimpleNamespace(method=method),
                    model_config=model,
                )
                ctx = {"self": SimpleNamespace(vllm_config=target, num_query_per_req=queries), "copy": copy}
                exec(snippet, ctx)
                draft = ctx["draft_config"]
                self.assertEqual(target.compilation_config.cudagraph_capture_sizes, sizes)
                self.assertIs(draft.model_config, model)
                if method == "dspark":
                    self.assertIsNot(draft.compilation_config, target.compilation_config)
                    self.assertEqual(
                        draft.compilation_config.cudagraph_capture_sizes, sorted(set(sizes + [queries, 16 * queries]))
                    )
                else:
                    self.assertIs(draft, target)

    def test_real_and_dummy_consumers_handle_absent_buffer(self):
        tree = ast.parse(self.runner)
        guards = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.Call)
            and isinstance(n.test.func, ast.Name)
            and n.test.func.id == "hasattr"
            and any(isinstance(x, ast.Constant) and x.value == "get_mtp_target_hidden_states" for x in n.test.args)
        ]
        self.assertEqual(len(guards), 2)
        for guard in guards:
            code = compile(ast.Module(body=[guard], type_ignores=[]), "<consumer>", "exec")
            for aux in [[], [object()]]:
                initial = object()
                ctx = {
                    "self": SimpleNamespace(model=SimpleNamespace(get_mtp_target_hidden_states=lambda: None)),
                    "aux_hidden_states": aux,
                    "spec_hidden_states": initial,
                }
                if aux:
                    exec(code, ctx)
                    self.assertIs(ctx["spec_hidden_states"], initial)
                else:
                    with self.assertRaisesRegex(RuntimeError, "requires aux hidden states"):
                        exec(code, ctx)

            class Buffer:
                def __getitem__(self, key):
                    return key.stop

            states = SimpleNamespace(shape=(7,), size=lambda dim: 7)
            ctx = {
                "self": SimpleNamespace(model=SimpleNamespace(get_mtp_target_hidden_states=lambda: Buffer())),
                "hidden_states": states,
                "draft_hidden_states": states,
                "aux_hidden_states": [],
            }
            exec(code, ctx)
            self.assertEqual(ctx["spec_hidden_states"], 7)


class TestMTPBufferAllocation(unittest.TestCase):
    def test_other_drafters_keep_the_buffer(self):
        repo = Path(__file__).resolve().parents[4]
        tree = ast.parse((repo / "vllm_ascend/models/deepseek_v41/model.py").read_text())
        assignment = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "needs_mtp_hidden_states" for t in n.targets)
        )
        expr = compile(ast.Expression(assignment.value), "<allocation>", "eval")
        for method, expected in [("dspark", False), ("eagle", True), ("dflash", True)]:
            spec = SimpleNamespace(method=method, use_eagle=lambda: True, uses_draft_model=lambda: True)
            self.assertIs(eval(expr, {"spec_config": spec}), expected)
        self.assertIs(eval(expr, {"spec_config": None}), False)


if __name__ == "__main__":
    unittest.main()
