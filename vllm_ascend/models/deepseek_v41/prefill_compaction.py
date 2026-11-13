# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DP-coordinated, terminal-only FFN row selection; all cache writes stay full."""

import numpy as np
import torch
import torch.distributed as dist
from vllm.config import CUDAGraphMode
from vllm.distributed import get_dp_group, get_tensor_model_parallel_rank, get_tp_group
from vllm.logger import logger

from vllm_ascend.ascend_forward_context import get_mrv2_in_profile_run
from vllm_ascend.models.deepseek_v41.consumer_compaction import supports_terminal_consumer
from vllm_ascend.utils import should_skip_allreduce_across_dp_group


def coordinated_terminal_count(eligible, benefits, counts, capacities, *, allow_dense_peers=False):
    """Every DP must agree, including decode-only and idle participants."""
    if not all(eligible) or not any(benefits):
        return 0
    padded = ((max(1, max(counts)) + 7) // 8) * 8
    if allow_dense_peers:
        return padded if padded < max(capacities) else 0
    return padded if padded <= min(capacities) else 0


def text_backbone(model):
    """Handle both the multimodal wrapper and a directly registered text model."""
    return getattr(model, "language_model", model).model


class TerminalPrefillCompaction:
    """Own call-scoped model inputs, CPU admission counts and a terminal-only plan."""

    def __init__(self, state):
        self.state = state
        self.backbone = text_backbone(state.model)
        config = state.vllm_config
        parallel = config.parallel_config
        if (
            parallel.tensor_parallel_size != 8
            or parallel.data_parallel_size != 2
            or not parallel.enable_expert_parallel
            or parallel.pipeline_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.enable_eplb
            or not parallel.use_sequence_parallel_moe
            or config.kv_transfer_config is not None
            or config.cache_config.swa_bounded_replay
            or should_skip_allreduce_across_dp_group(config)
        ):
            raise ValueError(
                "Terminal compaction requires TP8/DP2/EP16, FlashComm SP, non-PD and no CP/PP/EPLB or replay"
            )
        if (
            self.backbone._mtp_hidden_buffer is not None
            or not self.backbone.aux_hidden_state_layers
            or not supports_terminal_consumer(self.backbone.layers[-1].self_attn)
        ):
            raise ValueError("Terminal prefill compaction requires auxiliary-only DSpark and a final C1 consumer")
        self.sequence_parallel = parallel.use_sequence_parallel_moe
        self.tp_rank = get_tensor_model_parallel_rank()
        self.mlp = self.backbone.layers[-1].mlp
        runner = self.mlp.experts
        if (
            not runner.is_internal_router
            or runner.ascend_shared_experts is None
            or runner.routed_input_transform is not None
            or runner.routed_output_transform is not None
        ):
            raise ValueError("Terminal row compaction requires the plain internal-router/shared-expert MoE")
        runner.v41_terminal_compaction_owner = True
        runner.ascend_shared_experts.v41_terminal_compaction_owner = True
        self.mlp.prefill_compaction_plan = None
        self.verify = False
        self.oracle_indices = None
        self.oracle_used = None
        self.stats = dict(admitted_steps=0, executed_steps=0, original_rows=0, requested_rows=0, padded_rows=0)
        self.mlp.prefill_compaction_stats = self.stats
        self.mlp.prefill_compaction_oracle = self.verify_logits

    def prepare(self, batch, mode, for_capture):
        self.mlp.prefill_compaction_plan = None
        if mode != CUDAGraphMode.NONE or for_capture or get_mrv2_in_profile_run():
            return
        dummy = bool(getattr(batch, "is_dummy", False))
        worker = self.state.prompt_logprobs_worker
        scoring = worker is None or bool(getattr(worker, "token_id_scores", {}))
        if not scoring and not dummy:
            scoring = bool(worker.uses_prompt_logprobs[batch.idx_mapping_np[: batch.num_reqs]].any())
        count = 0 if dummy else batch.logits_indices.numel()
        benefit = not dummy and bool(batch.has_prefill) and batch.num_tokens > 256 and count < batch.num_tokens // 2
        dp = get_dp_group()
        local_rows = (
            (batch.num_tokens_after_padding + 7) // 8 if self.sequence_parallel else batch.num_tokens_after_padding
        )
        census = torch.zeros((5, dp.world_size), dtype=torch.int64)
        census[:, dp.rank_in_group] = torch.tensor(
            [
                not scoring,
                benefit,
                count,
                local_rows,
                self.verify,
            ],
            dtype=torch.int64,
        )
        # CPU-only scheduling data; no device reads or synchronizations. Every
        # rank joins NONE-mode admission even if its local batch only decodes.
        dist.all_reduce(census, group=dp.cpu_group)
        padded = coordinated_terminal_count(*census[:4].tolist(), allow_dense_peers=self.sequence_parallel)
        if not padded:
            return
        start = self.tp_rank * local_rows if self.sequence_parallel else 0
        if self.sequence_parallel and local_rows < padded:
            # AllToAllV allows different DP shard lengths. A small decode/idle
            # peer keeps its dense local FFN; it must not veto the prefill cut.
            padded = local_rows
            indices = torch.arange(padded, dtype=torch.int64, device=self.state.device)
            valid = (indices + start < batch.num_tokens) if not dummy else torch.zeros_like(indices, dtype=torch.bool)
        else:
            indices = torch.zeros(padded, dtype=torch.int64, device=self.state.device)
            if count:
                indices[:count].copy_(batch.logits_indices)
            valid = torch.arange(padded, device=self.state.device) < count
            if self.sequence_parallel:
                valid = valid & (indices >= start) & (indices < start + local_rows)
                indices = (indices - start).clamp(0, local_rows - 1)
        verify = bool(census[4].any())
        used = None
        if verify:
            self.oracle_indices = batch.logits_indices[:count]
            if count:
                done = (
                    batch.num_computed_prefill_tokens_np[: batch.num_reqs]
                    + batch.num_scheduled_tokens[: batch.num_reqs]
                    >= batch.prefill_len_np[: batch.num_reqs]
                )
                scored = np.repeat(done, np.diff(batch.cu_num_logits_np[: batch.num_reqs + 1]))
                self.oracle_used = torch.from_numpy(scored).to(self.state.device)
            else:
                self.oracle_used = torch.empty(0, dtype=torch.bool, device=self.state.device)
            used = valid
        self.mlp.prefill_compaction_plan = (indices, count, verify, valid, used)
        self.stats["admitted_steps"] += 1
        self.stats["original_rows"] += 0 if dummy else batch.num_tokens
        self.stats["requested_rows"] += count
        self.stats["padded_rows"] += padded
        if self.stats["admitted_steps"] == 1:
            logger.info(
                "Terminal DP2 FFN compaction admitted: original=%d selected=%d padded=%d; full KV/aux preserved",
                batch.num_tokens,
                count,
                padded,
            )

    def verify_logits(self, reference, compact, indices, used, residual, post, comb, pre, control=None):
        """Diagnostic-only full-vocabulary greedy decisions after the real HC/norm/head."""
        if self.oracle_indices.numel() == 0:
            return
        layer = self.backbone.layers[-1]
        tp = get_tp_group()
        states = []
        for ffn in (reference, compact) if control is None else (reference, compact, control):
            hidden = layer.hc_post(ffn.clone(), residual.clone(), post.clone(), comb.clone())
            hidden = layer.hc_collapse(hidden, pre)
            if self.sequence_parallel:
                hidden = tp.all_gather(hidden, dim=0)
            hidden = self.backbone.norm(hidden)
            states.append(hidden.index_select(0, self.oracle_indices))
        used = self.oracle_used
        text_model = getattr(self.state.model, "language_model", self.state.model)
        full_logits = text_model.compute_logits(states[0])
        cut_logits = text_model.compute_logits(states[1])
        assert full_logits is not None and cut_logits is not None
        if control is not None:
            control_logits = text_model.compute_logits(states[2])
            mismatches = (full_logits.argmax(-1) != control_logits.argmax(-1)) & used
            self.stats["full_ffn_self_greedy_mismatches"] = (
                self.stats.get("full_ffn_self_greedy_mismatches", 0) + mismatches.sum().item()
            )
        differences = full_logits.argmax(-1) != cut_logits.argmax(-1)
        self.stats["verified_logit_rows"] = self.stats.get("verified_logit_rows", 0) + used.sum().item()
        self.stats["greedy_mismatches"] = self.stats.get("greedy_mismatches", 0) + (differences & used).sum().item()
        error = torch.where(used.unsqueeze(1), (full_logits - cut_logits).abs(), 0).max().item()
        self.stats["max_logit_abs_difference"] = max(self.stats.get("max_logit_abs_difference", 0.0), error)
        eos = getattr(getattr(text_model, "config", None), "eos_token_id", None)
        eos_ids = list(eos) if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos])
        if eos_ids:
            mask = torch.tensor(eos_ids, dtype=torch.int64, device=full_logits.device)
            full_logits = full_logits.clone().index_fill_(1, mask, float("-inf"))
            cut_logits = cut_logits.clone().index_fill_(1, mask, float("-inf"))
            differences = full_logits.argmax(-1) != cut_logits.argmax(-1)
            ignored = (differences & used).sum().item()
            if control is not None:
                control_logits = control_logits.clone().index_fill_(1, mask, float("-inf"))
                self_ignored = ((full_logits.argmax(-1) != control_logits.argmax(-1)) & used).sum().item()
                self.stats["full_ffn_self_ignore_eos_greedy_mismatches"] = (
                    self.stats.get("full_ffn_self_ignore_eos_greedy_mismatches", 0) + self_ignored
                )
        else:
            ignored = (differences & used).sum().item()
        self.stats["ignore_eos_greedy_mismatches"] = self.stats.get("ignore_eos_greedy_mismatches", 0) + ignored

    def finish(self):
        self.mlp.prefill_compaction_plan = None
        self.oracle_indices = None
        self.oracle_used = None
