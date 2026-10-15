import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase

from vllm_ascend.distributed.device_communicators.npu_communicator import NPUCommunicator


def test_reduction_profile_only_selects_tensor_parallel_group(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_TP_BF16_ALLREDUCE_FP32", "1")
    monkeypatch.setattr(DeviceCommunicatorBase, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(torch.npu, "current_device", lambda: 0)

    tp = NPUCommunicator(None, unique_name="tp:0")
    dp = NPUCommunicator(None, unique_name="dp:0")

    assert tp._tp_fp32_all_reduce
    assert not dp._tp_fp32_all_reduce


def test_tp_bf16_reduction_accumulates_in_fp32_and_preserves_alias(monkeypatch):
    communicator = object.__new__(NPUCommunicator)
    communicator._tp_fp32_all_reduce = True
    communicator.device_group = object()
    calls = []

    def fake_all_reduce(tensor, group):
        calls.append((tensor.dtype, group))
        tensor.add_(0.5)

    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
    original = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    result = communicator.all_reduce(original)

    assert result is original
    assert result.tolist() == [1.5, 2.5]
    assert calls == [(torch.float32, communicator.device_group)]


def test_non_bf16_and_disabled_tp_use_base_implementation(monkeypatch):
    communicator = object.__new__(NPUCommunicator)
    communicator._tp_fp32_all_reduce = True
    seen = []

    def base_all_reduce(self, tensor):
        seen.append(tensor.dtype)
        return tensor

    monkeypatch.setattr(DeviceCommunicatorBase, "all_reduce", base_all_reduce)
    communicator.all_reduce(torch.tensor([1.0], dtype=torch.float32))
    communicator._tp_fp32_all_reduce = False
    communicator.all_reduce(torch.tensor([1.0], dtype=torch.bfloat16))

    assert seen == [torch.float32, torch.bfloat16]
