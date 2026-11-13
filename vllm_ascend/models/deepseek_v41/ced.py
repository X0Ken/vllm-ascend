# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit experimental CED roles; ordinary inference is unchanged by default."""

CED_SOURCE_LAYER = 20
CED_TARGET_LAYERS = 40
CED_REPLAY_TOKENS = 128
CED_TRANSFER_MARKER = 42


def received_state_replay_skip(replay_start, seq_lens, query_lens, used, received):
    """Skip only the received prefix's replay, retaining a new prompt tail."""
    first_position = (seq_lens - query_lens).clamp_min(0)
    skipped = (replay_start + CED_REPLAY_TOKENS - first_position).clamp_min(0).minimum(used)
    return skipped * ((replay_start > 0) & received)


def received_replay_positions(positions, replay_start, received):
    """Per-token guard works for mixed replay/decode and partial replay chunks."""
    return received & (replay_start > 0) & (positions >= replay_start) & (positions < replay_start + CED_REPLAY_TOKENS)


def replay_cache_origin(replay_start, seq_lens, query_lens):
    """Retain valid replay history, including ordinary decodes in mixed batches."""
    history_start = (seq_lens - query_lens - CED_REPLAY_TOKENS + 1).clamp_min(0)
    return history_start.maximum(replay_start)


def get_ced_role(vllm_config) -> str:
    role = (vllm_config.additional_config or {}).get("dsv41_ced_role", "off")
    if role not in ("off", "prefill", "decode"):
        raise ValueError("dsv41_ced_role must be off, prefill, or decode")
    if role == "off":
        return role
    parallel = vllm_config.parallel_config
    if (
        parallel.pipeline_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or parallel.data_parallel_size != 1
    ):
        raise ValueError("Experimental CED requires DP1 without PP/PCP/DCP")
    if role == "prefill" and vllm_config.speculative_config is not None:
        raise ValueError("CED P cannot supply the upper-layer states required by speculative decoding")
    transfer = vllm_config.kv_transfer_config
    expected_role = "kv_producer" if role == "prefill" else "kv_consumer"
    if (
        transfer is None
        or transfer.kv_connector != "MooncakeHybridConnector"
        or transfer.kv_role != expected_role
        or transfer.kv_connector_extra_config.get("named_cache_transfer", False) is not True
    ):
        raise ValueError("CED requires the matching MooncakeHybridConnector role and named_cache_transfer")
    if not vllm_config.cache_config.swa_bounded_replay:
        raise ValueError("CED requires SWA bounded replay")
    return role


def is_producer_weight(name: str) -> bool:
    """Never construct/load the upper decoder on a CED producer."""
    parts = name.split(".")
    if "layers" not in parts:
        return True
    index = parts.index("layers") + 1
    return int(parts[index]) <= CED_SOURCE_LAYER
