# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram projection and gate for BF16 and rotated Ascend checkpoints."""

import torch
from torch import nn
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.triton.engram_gate import fused_engram_gate

from .common import engram_gate


class AscendEngram(nn.Module):
    """Consume rows prepared by the v1 runner in the checkpoint residual basis."""

    def __init__(self, config, quant_config, prefix: str) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        shard_wkv = get_ascend_config().enable_dsv41_engram_wkv_tp
        self.shard_wkv = shard_wkv
        projection_cls = ColumnParallelLinear if shard_wkv else ReplicatedLinear
        self.wkv = projection_cls(
            (config.engram_max_ngram_size - 1) * config.engram_n_heads * config.engram_head_dim,
            (self.hc_mult + 1) * self.dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wkv",
            return_bias=False,
            **({"gather_output": True} if shard_wkv else {}),
        )
        self.q_weight = nn.Parameter(torch.empty(self.hc_mult, self.dim, dtype=torch.bfloat16))
        self.k_weight = nn.Parameter(torch.empty(self.hc_mult, self.dim, dtype=torch.bfloat16))

    def _project_rows(self, rows: torch.Tensor, sequence_parallel: bool) -> torch.Tensor:
        # Column shards must project identical tokens on all TP ranks. SP
        # inputs contain different token windows, so gather tokens first and
        # restore this rank's window after gathering the output columns.
        if self.shard_wkv and sequence_parallel:
            rows = tensor_model_parallel_all_gather(rows, dim=0)
            kv = self.wkv(rows)
            return kv.chunk(get_tensor_model_parallel_world_size(), dim=0)[get_tensor_model_parallel_rank()]
        return self.wkv(rows)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rows: torch.Tensor,
        token_mask: torch.Tensor,
        rotation: torch.Tensor | None,
        *,
        sequence_parallel: bool = False,
    ) -> torch.Tensor:
        kv = self._project_rows(rows, sequence_parallel)
        if (
            rotation is None
            and hidden_states.device.type == "npu"
            and get_current_hardware_profile().supports(HardwareCapability.ENGRAM_UNROTATED_GATE)
        ):
            return fused_engram_gate(hidden_states, kv, self.q_weight, self.k_weight, token_mask, self.eps)
        key, value = kv.split([self.hc_mult * self.dim, self.dim], -1)
        return engram_gate(
            hidden_states,
            key.view(hidden_states.shape[0], self.hc_mult, self.dim),
            value,
            self.q_weight.float() * self.k_weight.float(),
            rotation,
            token_mask,
            self.eps,
        )
