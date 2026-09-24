# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate DSpark logical query length from TP-aligned physical graph size.

Only trailing dummy requests may have a short query. Real requests retain all
six verification rows; the attention builder invalidates dummy cache slots.
"""

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

DSV41_K5_QUERY_LEN = 6
DSV41_COMPACT_SP_TP_SIZE = 8


class CompactSPGraphDispatcher(CudagraphDispatcher):
    def _create_padded_batch_descriptor(
        self, num_tokens: int, uniform_decode: bool, has_lora: bool, num_active_loras: int = 0
    ) -> BatchDescriptor:
        if not uniform_decode or not self.cudagraph_mode.has_mode(CUDAGraphMode.FULL):
            return super()._create_padded_batch_descriptor(num_tokens, uniform_decode, has_lora, num_active_loras)
        padded = self._bs_to_padded_graph_size[num_tokens]
        query_len = self.uniform_decode_query_len
        requests = (padded + query_len - 1) // query_len
        assert requests <= self.vllm_config.scheduler_config.max_num_seqs
        return BatchDescriptor(
            num_tokens=padded,
            num_reqs=requests,
            uniform=True,
            has_lora=has_lora,
            num_active_loras=num_active_loras,
        )
