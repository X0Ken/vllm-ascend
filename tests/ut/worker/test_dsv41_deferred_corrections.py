# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from vllm.config import CUDAGraphMode

from vllm_ascend.attention.dsa_v41 import AscendDSAV41MetadataBuilder
from vllm_ascend.worker.v1.model_runner import NPUModelRunner


def setup(enabled=True):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.use_compress = True
    runner._defer_v41_state_corrections = enabled
    runner.num_spec_tokens = 5
    runner.cache_config = SimpleNamespace(mamba_cache_mode="none")
    runner.input_batch = SimpleNamespace(
        req_ids=["a", "b"],
        num_reqs=2,
        prev_req_id_to_index={"a": 0, "b": 1},
        num_computed_tokens_cpu=np.array([106, 206], dtype=np.int32),
    )
    builder = AscendDSAV41MetadataBuilder.__new__(AscendDSAV41MetadataBuilder)
    runner.attn_groups = [[SimpleNamespace(get_metadata_builder=lambda: builder)]]
    runner.arange_np = np.arange(2)
    runner.query_pos = SimpleNamespace(np=np.tile(np.arange(6), 2))
    runner._dsa_positions_np_buf = np.full(12, -777, dtype=np.int32)
    schedule = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        scheduled_spec_decode_tokens={"a": [0] * 5, "b": [0] * 5},
        num_scheduled_tokens={"a": 6, "b": 6},
    )
    return runner, schedule


@pytest.mark.parametrize("enabled", [False, True])
def test_correction_is_retained_and_runs_after_submission_only_when_safe(enabled):
    runner, schedule = setup(enabled)
    order = []

    def correct():
        order.append("correct")
        runner.input_batch.num_computed_tokens_cpu -= np.array([5, 0])

    pending = runner._prepare_compressed_cpu_positions(schedule, CUDAGraphMode.FULL, np.array([6, 6]), 12, correct)
    if enabled:
        assert pending is correct and order == []
        np.testing.assert_array_equal(runner._dsa_positions_np_buf, np.full(12, -777))
        order.append("submit")
        pending()
        assert order == ["submit", "correct"]
    else:
        assert pending is None and order == ["correct"]
        np.testing.assert_array_equal(runner._dsa_positions_np_buf, np.r_[np.arange(101, 107), np.arange(206, 212)])
    np.testing.assert_array_equal(runner.input_batch.num_computed_tokens_cpu, [101, 206])


@pytest.mark.parametrize(
    "case",
    [
        "disabled",
        "eager",
        "piecewise",
        "new",
        "resumed",
        "unmapped",
        "partial",
        "query_width",
        "no_drafts",
        "no_requests",
        "legacy_builder",
        "empty_builders",
        "mamba",
    ],
)
def test_unsafe_batches_keep_early_correction_and_legacy_positions(case):
    runner, schedule = setup()
    mode = CUDAGraphMode.FULL
    if case == "disabled":
        runner._defer_v41_state_corrections = False
    elif case == "eager":
        mode = CUDAGraphMode.NONE
    elif case == "piecewise":
        mode = CUDAGraphMode.PIECEWISE
    elif case == "new":
        schedule.scheduled_new_reqs = [object()]
    elif case == "resumed":
        schedule.scheduled_cached_reqs.resumed_req_ids = {"a"}
    elif case == "unmapped":
        runner.input_batch.prev_req_id_to_index = {}
    elif case == "partial":
        schedule.scheduled_spec_decode_tokens["b"] = [0]
    elif case == "query_width":
        schedule.num_scheduled_tokens["b"] = 5
    elif case == "no_drafts":
        runner.num_spec_tokens = 0
    elif case == "no_requests":
        runner.input_batch.req_ids = []
    elif case == "legacy_builder":
        runner.attn_groups = [[SimpleNamespace(get_metadata_builder=lambda: object())]]
    elif case == "empty_builders":
        runner.attn_groups = [[]]
    elif case == "mamba":
        runner.cache_config.mamba_cache_mode = "align"
    callback = Mock()
    assert not runner._can_defer_v41_state_corrections(schedule, mode)
    assert runner._prepare_compressed_cpu_positions(schedule, mode, np.array([6, 6]), 12, callback) is None
    callback.assert_called_once_with()
    np.testing.assert_array_equal(runner._dsa_positions_np_buf, np.r_[np.arange(106, 112), np.arange(206, 212)])


def test_noncompressed_backend_keeps_existing_postforward_callback():
    runner, schedule = setup()
    runner.use_compress = False
    callback = Mock()
    assert (
        runner._prepare_compressed_cpu_positions(schedule, CUDAGraphMode.FULL, np.array([6, 6]), 12, callback)
        is callback
    )
    callback.assert_not_called()
