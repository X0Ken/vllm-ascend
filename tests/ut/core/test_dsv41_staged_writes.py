# SPDX-License-Identifier: Apache-2.0
"""Validate the installed pinned-core admission and block flush guards."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def table(groups=1):
    return SimpleNamespace(
        num_kv_cache_groups=groups,
        max_num_reqs=2,
        _num_blocks_dirty=True,
        block_tables=[
            SimpleNamespace(apply_write=Mock(), stage_write=Mock(), gpu=np.zeros((2, 4))) for _ in range(groups)
        ],
        num_blocks=SimpleNamespace(np=np.zeros((groups, 2), dtype=np.int32), copy_to_uva=Mock()),
        blocks_per_kv_block=[1] * groups,
        fused_writer=SimpleNamespace(apply=Mock()),
        block_table_ptrs=object(),
        block_table_strides=object(),
    )


@pytest.mark.parametrize("groups", [0, 1, 2])
def test_block_counts_flush_only_when_dirty(groups):
    t = table(groups)
    BlockTables.apply_staged_writes(t)
    BlockTables.apply_staged_writes(t)
    assert t.num_blocks.copy_to_uva.call_count == int(groups > 0)
    if groups:
        BlockTables.append_block_ids(t, 0, tuple([1, 2] for _ in range(groups)), True)
        BlockTables.apply_staged_writes(t)
        assert t.num_blocks.copy_to_uva.call_count == 2
        assert not t._num_blocks_dirty


def test_failed_block_count_copy_remains_dirty():
    t = table()
    t.num_blocks.copy_to_uva.side_effect = RuntimeError("copy failed")
    with pytest.raises(RuntimeError, match="copy failed"):
        BlockTables.apply_staged_writes(t)
    assert t._num_blocks_dirty
    t.num_blocks.copy_to_uva.side_effect = None
    BlockTables.apply_staged_writes(t)
    assert not t._num_blocks_dirty


def test_append_overflow_preserves_pending_flush():
    t = table()
    with pytest.raises(RuntimeError, match="capacity"):
        BlockTables.append_block_ids(t, 0, ([1, 2, 3, 4, 5],), True)
    assert t._num_blocks_dirty


def test_decode_without_admissions_skips_sampler_flush():
    runner = SimpleNamespace(sampler=SimpleNamespace(apply_staged_writes=Mock()))
    GPUModelRunner.add_requests(runner, SimpleNamespace(scheduled_new_reqs=[]))
    runner.sampler.apply_staged_writes.assert_not_called()


def test_admission_flushes_sampler_after_request_and_model_state():
    order = []
    flush = lambda name: Mock(side_effect=lambda: order.append(name))
    runner = SimpleNamespace(
        _remove_request=Mock(),
        req_states=SimpleNamespace(add_request=Mock(), req_id_to_index={"r": 0}, apply_staged_writes=flush("request")),
        adaptive_verification=None,
        pooling_runner=None,
        encoder_cache=None,
        model_state=SimpleNamespace(add_request=Mock(), apply_staged_writes=flush("model")),
        block_tables=SimpleNamespace(append_block_ids=Mock()),
        lora_state=SimpleNamespace(add_request=Mock()),
        is_last_pp_rank=True,
        sampler=SimpleNamespace(add_request=Mock(), apply_staged_writes=flush("sampler")),
        prompt_logprobs_worker=SimpleNamespace(add_request=Mock()),
    )
    request = SimpleNamespace(
        prefill_token_ids=[1, 2],
        req_id="r",
        prompt_len=2,
        sampling_params=SimpleNamespace(max_tokens=4),
        num_computed_tokens=0,
        block_ids=([0],),
        lora_request=None,
    )
    GPUModelRunner.add_requests(runner, SimpleNamespace(scheduled_new_reqs=[request]))
    assert order == ["request", "model", "sampler"]
