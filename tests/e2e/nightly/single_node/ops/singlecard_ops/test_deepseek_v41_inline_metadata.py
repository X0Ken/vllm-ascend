# SPDX-License-Identifier: Apache-2.0
"""Inline metadata must precede every eager forward and captured replay."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.dsa_v41 import AscendDSAV41MetadataBuilder
from vllm_ascend.worker.device_metadata import DeviceMetadataStage
from vllm_ascend.worker.v1 import model_runner as module


@pytest.mark.parametrize("batch", [1, 8, 32])
@pytest.mark.parametrize("full_graph", [False, True])
def test_inline_producer_refreshes_inputs_before_each_consumer(monkeypatch, batch, full_graph):
    torch.npu.set_device(0)
    runner = module.NPUModelRunner.__new__(module.NPUModelRunner)
    runner.device_metadata_executor = None
    runner.device_metadata_providers = None
    builder = AscendDSAV41MetadataBuilder.__new__(AscendDSAV41MetadataBuilder)
    builder._device_metadata_enabled = False
    builder._device_metadata_in_graph = False
    builder._device_metadata_tasks = ()
    builder.prepare_source_rope = lambda: None
    monkeypatch.setattr(module, "get_ascend_config", lambda: SimpleNamespace(enable_dsv41_inline_metadata=True))
    runner._initialize_device_metadata_providers({id(builder): builder})
    assert runner.device_metadata_executor is None
    source = torch.zeros(batch * 5, dtype=torch.int32, device="npu")
    metadata = torch.empty_like(source)
    output = torch.empty_like(source)
    pointer = metadata.data_ptr()

    def publish():
        builder._publish_task(
            {}, "metadata", metadata, DeviceMetadataStage.COMPRESSOR, lambda: torch.add(source, 7, out=metadata)
        )
        assert builder.take_device_metadata_tasks() == ()

    publish()
    output.copy_(metadata)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph() if full_graph else None
    if graph is not None:
        with torch.npu.graph(graph):
            output.copy_(metadata)
    snapshots = []
    for step in range(20):
        source.fill_(step)
        publish()
        if graph is not None:
            graph.replay()
        else:
            output.copy_(metadata)
        snapshots.append(output.clone())
        assert metadata.data_ptr() == pointer
    torch.npu.synchronize()
    for step, actual in enumerate(snapshots):
        torch.testing.assert_close(actual.cpu(), torch.full((batch * 5,), step + 7, dtype=torch.int32), rtol=0, atol=0)
