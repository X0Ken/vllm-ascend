# SPDX-License-Identifier: Apache-2.0
"""DSpark logical outputs retain storage through changing graph inputs."""

import pytest
import torch

from vllm_ascend.ops.triton.spec_decode.dspark_swa_indices import build_dspark_swa_indices_triton


@pytest.mark.parametrize("requests", [0, 1, 4, 8, 16, 32])
def test_swa_caller_outputs_and_guards(requests):
    tokens = requests * 5
    boundaries = torch.arange(requests + 1, dtype=torch.int32, device="npu") * 5
    lengths = torch.arange(requests, dtype=torch.int32, device="npu") * 133 + 127
    indices_owner = torch.full((tokens + 2, 1, 256), -777, dtype=torch.int32, device="npu")
    lengths_owner = torch.full((tokens + 2, 1), -777, dtype=torch.int32, device="npu")
    indices, visible = build_dspark_swa_indices_triton(
        boundaries,
        lengths,
        tokens,
        256,
        128,
        indices_output=indices_owner[1 : tokens + 1],
        lengths_output=lengths_owner[1 : tokens + 1],
    )
    if tokens:
        assert indices.data_ptr() == indices_owner[1:].data_ptr()
        assert visible.data_ptr() == lengths_owner[1:].data_ptr()
    for request, length in enumerate(lengths.cpu().tolist()):
        start = max(0, length - 5 - 128)
        expected = list(range(start, length)) + [-1] * (256 - length + start)
        assert indices[request * 5, 0].cpu().tolist() == expected
        assert visible[request * 5, 0].cpu().item() == length - start
    assert torch.all(indices_owner[[0, tokens + 1]] == -777)
    assert torch.all(lengths_owner[[0, tokens + 1]] == -777)


def test_swa_graph_replay_refreshes_zero_query_spans():
    boundaries = torch.tensor([0, 5, 5, 10], dtype=torch.int32, device="npu")
    lengths = torch.tensor([129, 8, 257], dtype=torch.int32, device="npu")
    indices = torch.empty((10, 1, 256), dtype=torch.int32, device="npu")
    visible = torch.empty((10, 1), dtype=torch.int32, device="npu")

    def build():
        build_dspark_swa_indices_triton(
            boundaries, lengths, 10, 256, 128, indices_output=indices, lengths_output=visible
        )

    build()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        build()
    boundaries.copy_(torch.tensor([0, 0, 5, 10], dtype=torch.int32, device="npu"))
    lengths.copy_(torch.tensor([1, 1000, 9000], dtype=torch.int32, device="npu"))
    graph.replay()
    torch.npu.synchronize()
    assert indices[0, 0, 0].cpu().item() == 867
    assert indices[5, 0, 0].cpu().item() == 8867
    assert visible.cpu().tolist() == [[133]] * 10
