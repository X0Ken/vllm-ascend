"""Limit prefill lookahead and preserve structured speculative sampling."""


def pending_prefill(queue, async_scheduling):
    return bool(async_scheduling and any(getattr(entry[1], "_v41_prefill_barrier", False) for entry in queue))


def prefill_ids(requests, async_scheduling):
    if not async_scheduling:
        return set()
    return {rid for rid, request in requests.items() if request.num_computed_tokens < request.num_prompt_tokens}


def tag_output(output, ids):
    output._v41_prefill_barrier = any(rid in ids for rid in output.num_scheduled_tokens)
    return output._v41_prefill_barrier


def backfill_structured_drafts(engine, output):
    """Resolve drafts before immediate sampling after the prefill barrier drains."""
    if not (
        engine.async_scheduling
        and engine.check_for_draft_tokens
        and output.has_structured_output_requests
        and output.scheduled_spec_decode_tokens
    ):
        return
    drafts = engine.model_executor.take_draft_token_ids()
    if drafts is not None:
        engine.scheduler.update_draft_token_ids_in_output(drafts, output)
