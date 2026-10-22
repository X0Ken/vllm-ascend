# SPDX-License-Identifier: Apache-2.0
"""Plan a bounded consumer suffix without dropping HC/DSpark dependencies."""

from dataclasses import dataclass


def supports_terminal_consumer(attention):
    """Keep packed-cache devices and producer layers on their original path."""
    role = getattr(attention, "role", None)
    return (
        role is not None
        and role.compress_ratio == 1
        and not role.is_kv_source
        and not role.is_index_source
        and getattr(attention, "packed_cache_ops", None) is None
    )


@dataclass(frozen=True)
class ConsumerCompactionPlan:
    boundary: int
    remaining_layers: int
    retained_tokens: int
    original_tokens: int
    eligible: bool
    reason: str


def plan_consumer_compaction(config, num_tokens, *, requires_full_hc, compact_metadata_supported, requested_suffix=1):
    layers = int(config["num_hidden_layers"])
    window = int(config["sliding_window"])
    if min(layers, window, requested_suffix) < 1 or num_tokens < 0:
        raise ValueError("Invalid model dimensions or requested suffix")
    # Conservatively finish all producer and feature layers before trimming.
    # This keeps every original DSpark feature row available to prefill KV.
    required = tuple(config["kv_source_layer_ids"]) + tuple(config["index_source_layer_ids"])
    required += tuple(config.get("engram_layer_ids", ()))
    required += tuple(int(layer) - 1 for layer in config.get("dspark_target_layer_ids", ()))
    required += (int(config.get("candidate_source_layer_id", -1)),)
    boundary = max(required, default=-1) + 1
    remaining = max(layers - boundary, 0)
    # Each local causal layer extends the dependency horizon by W-1. A fixed
    # W suffix at every layer is incorrect when those layers own separate SWA.
    retained = min(num_tokens, requested_suffix + remaining * (window - 1))
    reason = ""
    if requires_full_hc:
        reason = "Full-HC getter/buffer contract still requires all token rows"
    elif not compact_metadata_supported:
        reason = "Consumer-specific query/slot/SWA metadata rebuild is unavailable"
    elif not remaining or retained >= num_tokens:
        reason = "No removable suffix work after all producer/auxiliary dependencies"
    return ConsumerCompactionPlan(boundary, remaining, retained, num_tokens, not reason, reason)


def terminal_suffix_layout(query_start_loc, window):
    """Select a full local window from every prompt; short chunks fall back.

    A complete W-row suffix makes the requested last query independent of
    discarded local KV. Earlier suffix queries are bounded to retained rows
    and their outputs are not consumed by the terminal FFN.
    """
    if window < 1 or len(query_start_loc) < 2 or query_start_loc[0] != 0:
        raise ValueError("Invalid query boundaries or window")
    lengths = [int(end) - int(start) for start, end in zip(query_start_loc, query_start_loc[1:])]
    if any(length <= 0 for length in lengths):
        raise ValueError("Every compacted prompt must have a nonempty query")
    if min(lengths) < window or sum(lengths) <= window * len(lengths):
        return None
    indices = [row for end in query_start_loc[1:] for row in range(int(end) - window, int(end))]
    starts = [index * window for index in range(len(lengths) + 1)]
    outputs = [end - 1 for end in starts[1:]]
    return indices, starts, outputs
