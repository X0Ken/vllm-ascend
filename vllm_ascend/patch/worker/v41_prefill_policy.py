"""Limit lookahead only while a scheduled prefill batch is unresolved."""
def pending_prefill(queue, async_scheduling):
    return bool(async_scheduling and any(getattr(entry[1], '_v41_prefill_barrier', False) for entry in queue))
def prefill_ids(requests, async_scheduling):
    if not async_scheduling:return set()
    return {rid for rid,request in requests.items() if request.num_computed_tokens < request.num_prompt_tokens}
def tag_output(output, ids):
    output._v41_prefill_barrier = any(rid in ids for rid in output.num_scheduled_tokens)
    return output._v41_prefill_barrier
