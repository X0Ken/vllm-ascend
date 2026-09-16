import ctypes
import hashlib
import json
import os
import queue
import tempfile
import threading
import types
import unittest
from unittest.mock import MagicMock, patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    ReqMeta,
    infer_cache_key_config,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVCacheStoreSendingThread
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.transfer_audit import KVTransferAudit


class TestRankLocalKeys(unittest.TestCase):
    def model(self, sparse=True, mla=True, heads=1):
        return types.SimpleNamespace(
            model="org/model",
            use_mla=mla,
            hf_text_config=types.SimpleNamespace(**({"index_topk": 2048} if sparse else {})),
            get_total_num_kv_heads=lambda: heads,
            get_num_layers=lambda _: 46,
        )

    def worker(self, rank, sparse=True, mla=True, heads=1):
        worker = object.__new__(KVPoolWorker)
        worker.tp_rank, worker.tp_size = rank, 8
        worker.pcp_rank = worker.dcp_rank = worker.pp_rank = 0
        worker.pcp_size = worker.dcp_size = worker.pp_size = 1
        worker.use_mla, worker.use_sparse = mla, sparse
        worker.use_layerwise = False
        worker.kv_role = "kv_both"
        worker._extra_config = {}
        worker.group_uses_align_state = [False] * 6
        worker._init_key_head_config(self.model(sparse, mla, heads), None)
        return worker

    def database(self, worker):
        db = ChunkedTokenDatabase(
            [KeyMetadata(worker.model_name, worker.head_or_tp_rank, 0, 0, 0, g) for g in range(6)],
            [128] * 6,
            None,
            True,
            128,
        )
        db.set_group_buffers(
            {g: [1000] for g in range(6)},
            {g: [128] for g in range(6)},
            group_cache_families={g: "c1" for g in range(6)},
        )
        return db

    def test_save_load_lookup_agree_for_every_rank_and_group(self):
        workers = [self.worker(rank) for rank in range(8)]
        for group in range(6):
            saved = set()
            rank_keys = []
            for worker in workers:
                db = self.database(worker)
                chunks = list(
                    db.process_token_key_strings_with_block_ids(
                        256,
                        ["aa", "bb"],
                        [3, 5],
                        kv_cache_group_id=group,
                        shard_rank=worker.tp_rank % worker.put_step,
                        shard_size=worker.put_step,
                    )
                )
                self.assertEqual(len(chunks), 2)
                keys = [chunk[2] for chunk in chunks]
                self.assertEqual(
                    keys,
                    [
                        c[2]
                        for c in db.process_token_key_strings_with_block_ids(
                            256, ["aa", "bb"], [7, 9], kv_cache_group_id=group
                        )
                    ],
                )
                saved.update(keys)
                rank_keys.append(keys)
                self.assertEqual(worker.get_group_tp_size(group), 8)
            expanded = workers[0]._expand_lookup_keys_by_rank(rank_keys[0], group)
            self.assertEqual(set(expanded), saved)
            self.assertEqual(len(saved), 16)
            for missing_rank in range(8):
                available = saved - {rank_keys[missing_rank][1]}
                rows = [[int(key in available) for key in keys] for keys in rank_keys]
                self.assertEqual(KVPoolWorker.find_all_continuous_hit_positions(rows, [128, 256], 2, 256, 128), [128])

    def test_shared_mla_and_gqa_remain_sharded(self):
        for mla, heads, step in [(True, 1, 8), (False, 2, 4), (False, 8, 1)]:
            for rank in range(8):
                worker = self.worker(rank, sparse=False, mla=mla, heads=heads)
                self.assertEqual(worker.put_step, step)
                self.assertEqual(worker.head_or_tp_rank, rank // step)
                self.assertEqual(worker.model_name, "model")
                self.assertEqual(worker.get_group_tp_size(0), 8 // step)

    def test_namespace_separates_old_cache_and_different_tp_sizes(self):
        legacy = infer_cache_key_config(self.model(sparse=False), 8).model_name
        names = {infer_cache_key_config(self.model(), tp).model_name for tp in (1, 2, 4, 8)}
        self.assertEqual(len(names), 4)
        self.assertNotIn(legacy, names)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.torch.npu.Event")
    def test_sparse_save_finishes_reading_before_source_can_be_reused(self, event_factory):
        worker = self.worker(0)
        joining, reading, release, returned = (threading.Event() for _ in range(4))

        class ObservedQueue(queue.Queue):
            def join(self):
                joining.set()
                super().join()

        requests = ObservedQueue()
        source = bytearray(b"original")
        copied = []
        worker.kv_send_thread = MagicMock()
        worker.kv_send_thread.request_queue = requests
        worker.kv_send_thread.add_request.side_effect = requests.put
        meta = AscendConnectorMetadata(set(), set())
        meta.add_request(ReqMeta(req_id="save", token_len_chunk=128, can_save=True))

        def transfer():
            requests.get()
            reading.set()
            if release.wait(5):
                copied.append(bytes(source))
            requests.task_done()

        def forward():
            worker.wait_for_save(meta)
            source[:] = b"reused!!"
            returned.set()

        sender = threading.Thread(target=transfer, daemon=True)
        runner = threading.Thread(target=forward, daemon=True)
        sender.start()
        runner.start()
        try:
            self.assertTrue(reading.wait(5))
            self.assertTrue(joining.wait(5))
            self.assertFalse(returned.is_set())
        finally:
            release.set()
            sender.join(5)
            runner.join(5)
        self.assertTrue(returned.is_set())
        self.assertEqual(copied, [b"original"])

        shared = self.worker(0, sparse=False)
        shared.kv_send_thread = MagicMock()
        shared.wait_for_save(meta)
        shared.kv_send_thread.request_queue.join.assert_called_once()

    def test_scheduler_and_worker_generate_identical_keys(self):
        config = MagicMock()
        config.model_config = self.model()
        config.parallel_config.tensor_parallel_size = 8
        config.parallel_config.pipeline_parallel_size = 1
        config.parallel_config.prefill_context_parallel_size = 1
        config.parallel_config.decode_context_parallel_size = 1
        config.parallel_config.rank = 0
        config.cache_config.block_size = config.cache_config.hash_block_size = 128
        config.kv_transfer_config.kv_role = "kv_both"
        config.kv_transfer_config.kv_connector_extra_config = {"backend": "mooncake"}
        with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.importlib"):
            scheduler = KVPoolScheduler(config, False)
        scheduler.kv_cache_group_families = ["c1"] * 6
        for group in range(6):
            keys = scheduler._generate_store_query_keys(["aa"], kv_cache_group_id=group)
            expected = [
                list(self.database(self.worker(rank)).process_token_key_strings(128, ["aa"], kv_cache_group_id=group))[
                    0
                ][2]
                for rank in range(8)
            ]
            self.assertEqual(keys, [expected])

    def test_real_io_methods_preserve_rank_local_bytes_and_reject_partial_hits(self):
        payloads = {}

        def put(keys, addrs, sizes):
            for key, ptrs, lengths in zip(keys, addrs, sizes, strict=True):
                payloads[key] = [ctypes.string_at(ptr, size) for ptr, size in zip(ptrs, lengths, strict=True)]

        def get(keys, addrs, sizes):
            for key, ptrs, lengths in zip(keys, addrs, sizes, strict=True):
                for ptr, size, data in zip(ptrs, lengths, payloads[key], strict=True):
                    self.assertEqual(len(data), size)
                    ctypes.memmove(ptr, data, size)
            return [0] * len(keys)

        store = MagicMock()
        store.exists.side_effect = lambda keys: [int(key in payloads) for key in keys]
        store.put.side_effect, store.get.side_effect = put, get
        workers = []
        buffers = []
        for rank in range(8):
            worker = self.worker(rank)
            db = self.database(worker)
            rank_buffers = [ctypes.create_string_buffer(bytes([rank * 6 + g + 1]) * 256 + bytes(256)) for g in range(6)]
            buffers.append(rank_buffers)
            db.set_group_buffers(
                {g: [ctypes.addressof(buf)] for g, buf in enumerate(rank_buffers)}, {g: [128] for g in range(6)}
            )
            worker.token_database = db
            worker.m_store = store
            worker.grouped_block_size = [128] * 6
            worker.cache_transfer_granularity = 128
            worker.load_async = False
            worker.max_model_len = 256
            worker.cache_coordinator = None
            sender = KVCacheStoreSendingThread(
                store, db, [128] * 6, rank, tp_size=8, put_step=worker.put_step, kv_role="kv_both"
            )
            sender._handle_stored_request(
                ReqMeta(
                    req_id="save",
                    token_len_chunk=256,
                    block_ids_by_group=[[0, 1] for _ in range(6)],
                    block_hashes=["aa", "bb"],
                    kv_cache_group_ids=list(range(6)),
                )
            )
            workers.append(worker)
        self.assertEqual(len(payloads), 96)
        self.assertEqual(workers[0].lookup_scheduler(256, ["aa", "bb"], list(range(6))), 256)
        for group in range(6):
            key = list(workers[7].token_database.process_token_key_strings(256, ["aa", "bb"], kv_cache_group_id=group))[
                1
            ][2]
            removed = payloads.pop(key)
            self.assertEqual(workers[0].lookup_scheduler(256, ["aa", "bb"], list(range(6))), 128)
            payloads[key] = removed
        for rank, worker in enumerate(workers):
            req = ReqMeta(
                req_id="load",
                token_len_chunk=256,
                block_ids_by_group=[[2, 3] for _ in range(6)],
                block_hashes=["aa", "bb"],
                kv_cache_group_ids=list(range(6)),
                load_spec=LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=256, can_load=True, token_len=256),
            )
            meta = AscendConnectorMetadata(set(), set())
            meta.add_request(req)
            worker.start_load_kv(meta)
            for group, buf in enumerate(buffers[rank]):
                self.assertEqual(buf.raw[256:512], bytes([rank * 6 + group + 1]) * 256)


class TestTransferAudit(unittest.TestCase):
    def test_records_exact_bytes_and_propagates_copy_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("sys.modules", {"acl": types.SimpleNamespace(rt=MagicMock())}):
                audit = KVTransferAudit(directory, 3, 0, 0, 0)

            def copy(dst, capacity, src, length, kind):
                self.assertEqual(kind, 2)
                ctypes.memmove(dst, src, length)
                return 0

            audit._memcpy = copy
            buf = ctypes.create_string_buffer(b"abcdefgh")
            audit.record("save", "r", 2, ["key"], [0], [128], [7], [[ctypes.addressof(buf)]], [[8]])
            audit.record("load", "r", 2, ["key"], [0], [128], [9], [[ctypes.addressof(buf)]], [[8]])
            file = next(__import__("pathlib").Path(directory).iterdir())
            rows = [json.loads(line) for line in file.read_text().splitlines()]
            self.assertEqual(rows[0]["sha256"], [hashlib.sha256(b"abcdefgh").hexdigest()])
            self.assertEqual(rows[0]["sha256"], rows[1]["sha256"])
            self.assertEqual(rows[1]["block"], 9)
            self.assertEqual(file.stat().st_mode & 0o777, 0o600)
            audit._memcpy = lambda *args: 1
            with self.assertRaises(RuntimeError):
                audit._checksum(ctypes.addressof(buf), 8)
            os.close(audit._fd)
