"""Protect unresolved prefill state while allowing independent async decodes.

New prefills remain serialized behind the barrier; only unrelated D can pass.
The engine queue owns the request IDs until update_from_output consumes the
prefill result. Scheduler counters alone are unsafe: async scheduling advances
num_computed_tokens before the actual first sampled token has arrived.
"""

from vllm_ascend.patch.worker.v41_decode_first import is_prefill


def pending_prefill(queue, async_scheduling):
    return bool(async_scheduling and any(getattr(entry[1], "_v41_prefill_barrier", False) for entry in queue))


def prefill_ids(requests, async_scheduling):
    if not async_scheduling:
        return set()
    return {rid for rid, request in requests.items() if is_prefill(request)}


def tag_output(output, ids):
    output._v41_prefill_request_ids = frozenset(ids.intersection(output.num_scheduled_tokens))
    output._v41_prefill_barrier = bool(output._v41_prefill_request_ids)
    return output._v41_prefill_barrier


def blocked_prefills(queue, async_scheduling):
    if not async_scheduling:
        return frozenset()
    ids = set()
    for entry in queue:
        output = entry[1]
        if getattr(output, "_v41_prefill_barrier", False):
            tagged = getattr(output, "_v41_prefill_request_ids", None)
            if tagged is None:
                return None  # Preserve the global barrier for older untagged outputs.
            ids.update(tagged)
    return frozenset(ids)


def can_schedule(scheduler, queue, async_scheduling):
    if not pending_prefill(queue, async_scheduling):
        return True
    return bool(
        getattr(scheduler, "decode_first_enabled", False) and blocked_prefills(queue, async_scheduling) is not None
    )


def schedule(engine, queue):
    scheduler = engine.scheduler
    if not getattr(scheduler, "decode_first_enabled", False):
        return scheduler.schedule(engine._should_throttle_prefills())
    blocked = blocked_prefills(queue, engine.async_scheduling)
    assert blocked is not None
    return scheduler.schedule(
        engine._should_throttle_prefills(),
        blocked_request_ids=blocked,
        block_prefills=bool(blocked),
    )


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
