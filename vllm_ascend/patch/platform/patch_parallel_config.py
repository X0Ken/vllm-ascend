# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.parallel import ParallelConfig


def _use_sequence_parallel_moe(self: ParallelConfig) -> bool:
    """Use the p8 MoE layout for DP=1 to avoid small-batch SP padding."""
    return (
        self.all2all_backend
        in (
            "allgather_reducescatter",
            "deepep_high_throughput",
            "deepep_low_latency",
            "deepep_v2",
            "flashinfer_nvlink_one_sided",
            "mori_high_throughput",
            "mori_low_latency",
            "nixl_ep",
        )
        and self.data_parallel_size > 1
        and self.enable_expert_parallel
        and self.tensor_parallel_size > 1
    )


# Keep upstream's DP>1 requirement in the V4.1 performance branch. For DP=1,
# SP forces K5 target graphs to multiples of lcm(6, TP), increasing c1-c8 work.
ParallelConfig.use_sequence_parallel_moe = property(_use_sequence_parallel_moe)
