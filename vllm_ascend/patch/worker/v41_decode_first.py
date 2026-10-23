# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step D/P budgets used by the pinned vLLM 0.31 scheduler patch.

The scheduler owns this state for one schedule() call. Draft workspace remains
charged to upstream's input_budget, separately from scheduled target rows.
Remove with v41_decode_first.patch once upstream exposes phase/budget hooks.
"""

from pydantic import TypeAdapter


def decode_first_config(vllm_config):
    additional = vllm_config.additional_config or {}
    return TypeAdapter(bool).validate_python(additional.get("enable_dsv41_decode_first", False))


def is_prefill(request, num_computed_tokens=None):
    """Include prompt work and recomputation of preempted output history."""
    computed = request.num_computed_tokens if num_computed_tokens is None else num_computed_tokens
    return computed < request.num_prompt_tokens or computed < getattr(request, "num_tokens", 0) - 1


class PhaseUsage:
    def __init__(self):
        self.prefills = {}
        self.decodes = set()

    def record(self, request_id, tokens, prefill):
        if prefill:
            self.prefills[request_id] = tokens
        else:
            self.decodes.add(request_id)

    def remove(self, request_id):
        self.prefills.pop(request_id, None)
        self.decodes.discard(request_id)

    def snapshot(self):
        return {
            "decode_request_ids": sorted(self.decodes),
            "prefill_tokens": sum(self.prefills.values()),
        }
