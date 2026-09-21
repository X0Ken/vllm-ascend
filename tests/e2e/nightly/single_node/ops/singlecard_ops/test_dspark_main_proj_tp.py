# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun --nproc-per-node=2 (or 8) -m pytest -s this_file.py.

The test initializes HCCL and must run only after the service releases the NPUs.
It uses synthetic BF16 weights with the checkpoint's [4096, 12288] shape.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import adapt_patch, enable_custom_op, register_ascend_customop

# Keep this file runnable from /tmp without the repository's conftest. Loading
# model/attention modules before these hooks can re-enter DeviceOperator during
# its initialization. These are the same idempotent hooks as tests/ut/conftest.
adapt_patch()
adapt_patch(True)
register_ascend_customop()

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed import init_distributed_environment  # noqa: E402
from vllm.distributed.parallel_state import (  # noqa: E402
    destroy_distributed_environment,
    destroy_model_parallel,
    initialize_model_parallel,
)

import vllm_ascend.models.deepseek_v4_dspark as dspark  # noqa: E402
from vllm_ascend.ops.linear import (  # noqa: E402
    AscendColumnParallelLinear,
    AscendReplicatedLinear,
    AscendUnquantizedLinearMethod,
)

pytestmark = pytest.mark.skipif(
    int(os.getenv("WORLD_SIZE", "1")) < 2, reason="Launch with torchrun on 2 or 8 free NPUs"
)


@pytest.fixture(scope="module")
def tp_environment():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(local_rank)
    if not enable_custom_op():
        raise RuntimeError("The installed Ascend custom ops are required for this NPU test")
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world))
    with set_current_vllm_config(config):
        init_distributed_environment(world_size=world, rank=rank, local_rank=local_rank, backend="hccl")
        initialize_model_parallel(tensor_model_parallel_size=world, backend="hccl")
        try:
            yield rank, world, torch.device(f"npu:{local_rank}")
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


@pytest.mark.parametrize("hidden", [256, 4096])
@pytest.mark.parametrize("nz_mode", [0, 2])
def test_main_projection_tp_matches_replicated_after_loading(tp_environment, hidden, nz_mode):
    rank, world, device = tp_environment
    generator = torch.Generator(device="cpu").manual_seed(142)
    full_weight = (torch.randn(hidden, hidden * 3, generator=generator) / (hidden * 3) ** 0.5).bfloat16()
    prefix = "model.layers.61.main_proj"
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with (
            patch.object(dspark, "ColumnParallelLinear", AscendColumnParallelLinear),
            patch.object(dspark, "ReplicatedLinear", AscendReplicatedLinear),
            patch.object(dspark, "get_ascend_config", return_value=SimpleNamespace(dspark_main_proj_tp=True)),
            patch.object(dspark, "enable_sp", return_value=False),
            patch.object(dspark, "enable_sp_by_pass", return_value=False),
            patch("vllm_ascend.ops.linear_op.enable_sp", return_value=False),
            patch("vllm_ascend.ops.linear_op.enable_dsa_cp", return_value=False),
            patch("vllm_ascend.utils.get_ascend_config", return_value=SimpleNamespace(weight_nz_mode=nz_mode)),
            torch.device(device),
        ):
            candidate = dspark._build_dspark_main_proj(
                SimpleNamespace(hidden_size=hidden, dspark_target_layer_ids=[3, 17, 31]), prefix
            )
            baseline = AscendReplicatedLinear(hidden * 3, hidden, bias=False, return_bias=False, prefix=prefix)
            assert isinstance(candidate, AscendColumnParallelLinear)
            assert isinstance(candidate.quant_method, AscendUnquantizedLinearMethod)
            assert candidate.custom_op is None
            assert candidate.bias is None and candidate.gather_output
            assert candidate.weight.shape == (hidden // world, hidden * 3)
            candidate.weight.weight_loader(candidate.weight, full_weight)
            baseline.weight.weight_loader(baseline.weight, full_weight)
            torch.testing.assert_close(candidate.weight.cpu(), full_weight.chunk(world, dim=0)[rank])
            for layer in (baseline, candidate):
                layer.quant_method.process_weights_after_loading(layer)
            for tokens in (1, 7, 32, 64):
                x = torch.randn(tokens, hidden * 3, generator=generator, device="cpu").bfloat16().to(device)
                with torch.inference_mode():
                    expected, actual = baseline(x), candidate(x)
                assert actual.shape == expected.shape == (tokens, hidden)
                torch.testing.assert_close(actual, expected, rtol=0.016, atol=0.016)
                # RMSNorm must operate on the gathered full hidden dimension.
                expected_norm = expected.float() * torch.rsqrt(expected.float().square().mean(-1, keepdim=True) + 1e-6)
                actual_norm = actual.float() * torch.rsqrt(actual.float().square().mean(-1, keepdim=True) + 1e-6)
                torch.testing.assert_close(actual_norm, expected_norm, rtol=0.02, atol=0.02)
                if rank == 0:
                    print(
                        f"TP={world} H={hidden} T={tokens} NZ={nz_mode} "
                        f"max_abs={float((actual.float() - expected.float()).abs().max()):.6g}"
                    )
    finally:
        torch.set_default_dtype(old_dtype)
