# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression for ADXL's check-then-bind race and initialization failures."""

import concurrent.futures
import importlib.util
import multiprocessing
import pathlib
import sys
import time
import types
import unittest
from unittest.mock import patch

SOURCE = (
    pathlib.Path(__file__).resolve().parents[4]
    / "vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py"
)


def load():
    spec = importlib.util.spec_from_file_location("transfer_init", SOURCE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def child(barrier, active, peak, queue):
    class Engine:
        def initialize(self, *args):
            with active.get_lock():
                active.value += 1
                peak.value = max(peak.value, active.value)
            time.sleep(0.08)
            with active.get_lock():
                active.value -= 1
            return 0

    sys.modules["mooncake.engine"] = types.SimpleNamespace(TransferEngine=Engine)
    m = load()
    barrier.wait(timeout=20)
    try:
        queue.put(isinstance(m.GlobalTE().get_transfer_engine("127.0.0.1", None), Engine))
    except Exception as e:
        queue.put(repr(e))


class TransferEngineTest(unittest.TestCase):
    def setUp(self):
        modules = patch.dict(sys.modules)
        modules.start()
        self.addCleanup(modules.stop)

    def test_concurrent_processes_serialize_check_and_bind(self):
        ctx = multiprocessing.get_context("spawn")
        barrier = ctx.Barrier(4)
        active = ctx.Value("i", 0)
        peak = ctx.Value("i", 0)
        queue = ctx.Queue()
        children = [ctx.Process(target=child, args=(barrier, active, peak, queue)) for _ in range(4)]
        for p in children:
            p.start()
        results = [queue.get(timeout=25) for _ in children]
        for p in children:
            p.join(25)
            self.assertEqual(p.exitcode, 0)
        self.assertEqual(results, [True] * 4)
        self.assertEqual(peak.value, 1)

    def test_failed_engine_is_not_published(self):
        m = load()
        calls = []

        class Engine:
            def initialize(self, *args):
                calls.append(args)
                return -1 if len(calls) == 1 else 0

        sys.modules["mooncake.engine"] = types.SimpleNamespace(TransferEngine=Engine)
        g = m.GlobalTE()
        with self.assertRaises(RuntimeError):
            g.get_transfer_engine("host", None)
        self.assertIsNone(g.transfer_engine)
        good = g.get_transfer_engine("host", None)
        self.assertIs(g.get_transfer_engine("host", None), good)
        self.assertEqual(len(calls), 2)

    def test_threads_publish_one_engine(self):
        m = load()
        calls = []

        class Engine:
            def initialize(self, *args):
                calls.append(args)
                time.sleep(0.02)
                return 0

        sys.modules["mooncake.engine"] = types.SimpleNamespace(TransferEngine=Engine)
        g = m.GlobalTE()
        with concurrent.futures.ThreadPoolExecutor(8) as ex:
            values = list(ex.map(lambda _: g.get_transfer_engine("host", None), range(8)))
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(v is values[0] for v in values))


if __name__ == "__main__":
    unittest.main()
