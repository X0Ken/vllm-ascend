# SPDX-License-Identifier: Apache-2.0
"""Inline V4.1 producers must finish staging before model replay."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.attention.dsa_v41 import AscendDSAV41MetadataBuilder
from vllm_ascend.worker import model_runner_v1 as module
from vllm_ascend.worker.device_metadata import DeviceMetadataStage


@pytest.mark.parametrize("inline", [False, True])
def test_target_metadata_uses_selected_stream_owner(inline):
    runner = module.NPUModelRunner.__new__(module.NPUModelRunner)
    runner.device_metadata_executor = None
    runner.device_metadata_providers = None
    builder = AscendDSAV41MetadataBuilder.__new__(AscendDSAV41MetadataBuilder)
    builder._device_metadata_enabled = False
    builder._device_metadata_in_graph = False
    builder._device_metadata_tasks = ()
    builder.prepare_source_rope = MagicMock()
    providers = {id(builder): builder}
    with (
        patch.object(module, "get_ascend_config", return_value=SimpleNamespace(enable_dsv41_inline_metadata=inline)),
        patch.object(module, "DeviceMetadataExecutor") as executor,
    ):
        runner._initialize_device_metadata_providers(providers)
    builder.prepare_source_rope.assert_called_once_with()
    output = torch.zeros(3)
    builder._publish_task({}, "producer", output, DeviceMetadataStage.COMPRESSOR, lambda: output.fill_(7))
    if inline:
        executor.assert_not_called()
        assert runner.device_metadata_executor is None
        assert runner.device_metadata_providers is None
        assert builder.take_device_metadata_tasks() == ()
        torch.testing.assert_close(output, torch.full((3,), 7.0))
    else:
        executor.assert_called_once_with()
        assert runner.device_metadata_providers is providers
        tasks = builder.take_device_metadata_tasks()
        assert len(tasks) == 1
        torch.testing.assert_close(output, torch.zeros(3))
        tasks[0].run()
        torch.testing.assert_close(output, torch.full((3,), 7.0))


def test_inline_rejects_non_v41_producers_before_initializing_any():
    runner = module.NPUModelRunner.__new__(module.NPUModelRunner)
    with (
        patch.object(module, "get_ascend_config", return_value=SimpleNamespace(enable_dsv41_inline_metadata=True)),
        pytest.raises(ValueError, match="V4.1 builders exclusively"),
    ):
        runner._initialize_device_metadata_providers({1: MagicMock()})
