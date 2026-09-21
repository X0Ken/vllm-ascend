# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU resource-lifetime checks; real lookup equality is tested on NPU."""

import ctypes
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def module():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/engram_uva.py"
    spec = importlib.util.spec_from_file_location("engram_uva_test", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Runtime:
    def __init__(self, failure=None):
        self.failure = failure
        self.allocations = {}
        self.calls = []

    def aclrtMallocHost(self, output, size, flag):
        self.calls.append("allocate")
        if self.failure == "allocate":
            return 1
        memory = ctypes.create_string_buffer(size)
        address = ctypes.addressof(memory)
        self.allocations[address] = memory
        output._obj.value = address
        return 0

    def aclrtHostRegisterV2(self, pointer, size, flags):
        self.calls.append("register")
        return int(self.failure == "register")

    def aclrtHostGetDevicePointer(self, pointer, output, flags):
        self.calls.append("map")
        output._obj.value = pointer.value
        return int(self.failure == "map")

    def aclrtHostUnregister(self, pointer):
        self.calls.append("unregister")
        return 0

    def aclrtFreeHost(self, pointer):
        self.calls.append("free")
        del self.allocations[pointer.value]
        return 0


def setup_runtime(module, monkeypatch, failure=None):
    runtime = Runtime(failure)
    monkeypatch.setattr(module, "_host_library", lambda: runtime)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=lambda device: None), raising=False)
    return runtime


@pytest.mark.parametrize("failure", ["allocate", "register", "map"])
def test_partial_initialization_releases_host_allocation(module, monkeypatch, failure):
    runtime = setup_runtime(module, monkeypatch, failure)
    with pytest.raises(RuntimeError):
        module.HostUvaBuffer((16, 32), torch.int8, "cpu")
    assert not runtime.allocations
    if failure == "map":
        assert runtime.calls[-2:] == ["unregister", "free"]


def test_close_is_idempotent_and_chunk_addresses_follow_row_bytes(module, monkeypatch):
    runtime = setup_runtime(module, monkeypatch)
    monkeypatch.setattr(module, "CHUNK_ROWS", 4)
    buffer = module.HostUvaBuffer((9, 32), torch.float32, "cpu")
    assert buffer.ptrs.tolist() == [buffer.pointer.value + n * 32 * 4 for n in [0, 4, 8]]
    buffer.tensor.fill_(7)
    assert torch.all(buffer.tensor == 7)
    buffer.close()
    buffer.close()
    assert runtime.calls.count("free") == 1
    assert not runtime.allocations


def test_pointer_upload_failure_releases_mapping(module, monkeypatch):
    runtime = setup_runtime(module, monkeypatch)

    def fail(*args, **kwargs):
        raise RuntimeError("upload failed")

    monkeypatch.setattr(torch, "tensor", fail)
    with pytest.raises(RuntimeError, match="upload failed"):
        module.HostUvaBuffer((16, 32), torch.int8, "cpu")
    assert not runtime.allocations
    assert runtime.calls[-2:] == ["unregister", "free"]
