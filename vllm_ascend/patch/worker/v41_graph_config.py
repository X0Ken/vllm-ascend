# SPDX-License-Identifier: Apache-2.0
"""Keep DSpark draft graph sizes independent of target graph tuning."""

LEGACY_DRAFT_CAPTURE_REQUESTS = 16


def draft_capture_sizes(vllm_config, query_tokens: int) -> list[int]:
    additional = vllm_config.additional_config or {}
    sizes = additional.get("dspark_draft_graph_capture_sizes")
    if sizes is None:
        sizes = vllm_config.compilation_config.cudagraph_capture_sizes or []
    elif not isinstance(sizes, list) or not sizes or any(type(size) is not int or size <= 0 for size in sizes):
        raise ValueError("dspark_draft_graph_capture_sizes must be a nonempty list of positive integer token counts")
    return sorted(set(sizes + [query_tokens, LEGACY_DRAFT_CAPTURE_REQUESTS * query_tokens]))
