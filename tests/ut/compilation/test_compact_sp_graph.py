# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from vllm.config import CUDAGraphMode

from vllm_ascend.compilation.compact_sp_graph import CompactSPGraphDispatcher
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


@pytest.fixture
def dispatcher():
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            max_cudagraph_capture_size=192,
            cudagraph_capture_sizes=[8] + list(range(24, 193, 24)),
            compile_sizes=[],
        ),
        num_speculative_tokens=5,
        scheduler_config=SimpleNamespace(max_num_seqs=32),
        lora_config=None,
    )
    dispatcher = CompactSPGraphDispatcher(config)
    dispatcher.initialize_cudagraph_keys(CUDAGraphMode.FULL_DECODE_ONLY, 6)
    return dispatcher


@pytest.mark.parametrize("requests", range(1, 33))
def test_compact_keys_and_real_prefix_preservation(dispatcher, requests):
    input_physical = ((requests * 6 + 7) // 8) * 8
    physical = 8 if requests == 1 else ((requests * 6 + 23) // 24) * 24
    mode, descriptor = dispatcher.dispatch(input_physical, uniform_decode=True)
    assert mode == CUDAGraphMode.FULL
    assert descriptor.num_tokens == physical
    assert descriptor.num_reqs == (physical + 5) // 6
    assert descriptor in dispatcher.cudagraph_keys[CUDAGraphMode.FULL]

    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
        uniform_decode_query_len=6,
        arange_np=np.arange(513, dtype=np.int32),
    )
    # DP agreement can enlarge an otherwise smaller local batch. Only dummy
    # suffix requests may change; existing request offsets must survive.
    for local_requests in range(requests + 1):
        original = np.arange(local_requests + 1, dtype=np.int32) * 6
        offsets = np.full(33, -123, dtype=np.int32)
        offsets[: local_requests + 1] = original
        buffer = SimpleNamespace(np=offsets, copy_to_gpu=MagicMock())
        with patch(
            "vllm_ascend.worker.model_runner_v1.get_ascend_config",
            return_value=SimpleNamespace(enable_dsv41_compact_sp_graph=True),
        ):
            padded_requests = NPUModelRunner._pad_query_start_loc_for_fia(
                runner, buffer, physical, requests, local_requests, mode, descriptor.num_reqs
            )
        assert padded_requests == descriptor.num_reqs
        np.testing.assert_array_equal(offsets[: local_requests + 1], original)
        assert offsets[padded_requests] == physical
        lengths = np.diff(offsets[: padded_requests + 1])
        assert np.all(lengths > 0) and np.all(lengths <= 6)
        assert np.all(offsets[padded_requests + 1 :] == -123)
        buffer.copy_to_gpu.assert_called_once_with()


@pytest.mark.parametrize("tokens", [1, 8, 24, 48, 192])
def test_mixed_batch_keeps_eager_fallback(dispatcher, tokens):
    assert dispatcher.dispatch(tokens, uniform_decode=False)[0] == CUDAGraphMode.NONE


def test_oversize_keeps_eager_fallback(dispatcher):
    assert dispatcher.dispatch(193, uniform_decode=True)[0] == CUDAGraphMode.NONE


def test_default_padding_path_is_unchanged():
    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
        uniform_decode_query_len=6,
        arange_np=np.arange(513, dtype=np.int32),
    )
    offsets = np.zeros(33, dtype=np.int32)
    offsets[1] = 6
    buffer = SimpleNamespace(np=offsets, copy_to_gpu=MagicMock())
    with patch(
        "vllm_ascend.worker.model_runner_v1.get_ascend_config",
        return_value=SimpleNamespace(enable_dsv41_compact_sp_graph=False),
    ):
        count = NPUModelRunner._pad_query_start_loc_for_fia(runner, buffer, 24, 4, 1, CUDAGraphMode.FULL, 4)
    assert count == 4
    np.testing.assert_array_equal(offsets[:5], [0, 6, 12, 18, 24])


def test_multi_request_keys_match_p6(dispatcher):
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

    config = dispatcher.vllm_config
    config.compilation_config.cudagraph_capture_sizes = list(range(24, 193, 24))
    original = CudagraphDispatcher(config)
    original.initialize_cudagraph_keys(CUDAGraphMode.FULL_DECODE_ONLY, 6)
    for requests in range(2, 33):
        incoming = ((requests * 6 + 7) // 8) * 8
        assert dispatcher.dispatch(incoming, uniform_decode=True) == original.dispatch(incoming, uniform_decode=True)


def test_aligned_graphs_keep_original_padding():
    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
        uniform_decode_query_len=6,
        arange_np=np.arange(513, dtype=np.int32),
    )
    offsets = np.zeros(33, dtype=np.int32)
    offsets[:3] = [0, 6, 12]
    buffer = SimpleNamespace(np=offsets, copy_to_gpu=MagicMock())
    with (
        patch(
            "vllm_ascend.worker.model_runner_v1.get_ascend_config",
            return_value=SimpleNamespace(enable_dsv41_compact_sp_graph=True),
        ),
        patch("vllm_ascend.worker.model_runner_v1.np.minimum", side_effect=AssertionError("Unexpected compact path")),
    ):
        count = NPUModelRunner._pad_query_start_loc_for_fia(runner, buffer, 24, 4, 2, CUDAGraphMode.FULL, 4)
    assert count == 4
    np.testing.assert_array_equal(offsets[:5], [0, 6, 12, 18, 24])
    buffer.copy_to_gpu.assert_called_once_with()
