# SPDX-License-Identifier: Apache-2.0
"""Real D2H count events and GPU position correction with reordered batches."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401
from vllm.config import CUDAGraphMode

from vllm_ascend.attention.dsa_v41 import AscendDSAV41MetadataBuilder
from vllm_ascend.worker.spec_decode_utils import update_num_computed_tokens_for_batch_change
from vllm_ascend.worker.v1.model_runner import NPUModelRunner


@pytest.mark.parametrize("batch", [1, 4, 8])
@pytest.mark.parametrize("enabled", [False, True])
@torch.inference_mode()
def test_deferred_cpu_counts_match_device_positions_after_rejections(batch, enabled):
    torch.npu.set_device(0)
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.use_compress = True
    runner._defer_v41_state_corrections = enabled
    runner.num_spec_tokens = 5
    runner.cache_config = SimpleNamespace(mamba_cache_mode="none")
    builder = AscendDSAV41MetadataBuilder.__new__(AscendDSAV41MetadataBuilder)
    runner.attn_groups = [[SimpleNamespace(get_metadata_builder=lambda: builder)]]
    runner.arange_np = np.arange(batch)
    runner.query_pos = SimpleNamespace(np=np.tile(np.arange(6), batch))
    runner._dsa_positions_np_buf = np.full(batch * 6, -777, dtype=np.int32)
    runner.valid_sampled_token_count_cpu = torch.empty(batch, dtype=torch.int32, pin_memory=True)
    runner.valid_sampled_token_count_event = torch.npu.Event()
    copy_stream = torch.npu.Stream()
    for iteration in range(4):
        previous = np.arange(batch, dtype=np.int32) * 100 + 512 + iteration * 16
        mapping = np.arange(batch, dtype=np.int32)[::-1].copy()
        counts = np.resize(np.roll(np.array([1, 3, 6], dtype=np.int32), iteration), batch)
        ids = [str(i) for i in mapping]
        runner.input_batch = SimpleNamespace(
            req_ids=ids,
            num_reqs=batch,
            prev_req_id_to_index={str(i): i for i in range(batch)},
            num_computed_tokens_cpu=previous[mapping] + 6,
            prev_sampled_token_ids=torch.empty(batch, 6, dtype=torch.int32, device="npu"),
        )
        schedule = SimpleNamespace(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
            scheduled_spec_decode_tokens={rid: [0] * 5 for rid in ids},
            num_scheduled_tokens={rid: 6 for rid in ids},
        )
        count_gpu = torch.tensor(counts, device="npu")
        producer_stream = torch.npu.current_stream()
        with torch.npu.stream(copy_stream):
            copy_stream.wait_stream(producer_stream)
            runner.valid_sampled_token_count_cpu.copy_(count_gpu, non_blocking=True)
            runner.valid_sampled_token_count_event.record(copy_stream)
        order = []

        def correct(mapping=mapping, order=order):
            actual = runner._get_valid_sampled_token_count()
            runner.input_batch.num_computed_tokens_cpu -= 6 - np.array(actual, dtype=np.int32)[mapping]
            order.append("correct")

        gpu_positions = torch.tensor(previous, device="npu")
        cpu_upper = torch.tensor(runner.input_batch.num_computed_tokens_cpu.copy(), device="npu")
        callback = runner._prepare_compressed_cpu_positions(
            schedule, CUDAGraphMode.FULL, np.full(batch, 6), batch * 6, correct
        )
        assert order == ([] if enabled else ["correct"])
        accepted = torch.ones(batch, dtype=torch.int32, device="npu")
        update_num_computed_tokens_for_batch_change(
            gpu_positions,
            accepted,
            torch.tensor(mapping, device="npu"),
            count_gpu,
            torch.full((batch,), 5, dtype=torch.int32, device="npu"),
            cpu_upper,
        )
        output = gpu_positions[:, None] + torch.arange(6, device="npu")[None, :]
        order.append("submit")
        if callback is not None:
            callback()
        assert order == (["submit", "correct"] if enabled else ["correct", "submit"])
        expected = previous[mapping] + counts[mapping]
        np.testing.assert_array_equal(runner.input_batch.num_computed_tokens_cpu, expected)
        torch.testing.assert_close(
            output.cpu(), torch.tensor(expected[:, None] + np.arange(6)[None, :]), rtol=0, atol=0
        )
        torch.testing.assert_close(accepted.cpu(), torch.tensor(counts[mapping]), rtol=0, atol=0)
