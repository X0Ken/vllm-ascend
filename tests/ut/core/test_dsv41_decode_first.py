# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual pinned scheduler, allocations, and async request state."""

from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from tests.ut.core.test_dyntra_lb_scheduler import create_dyntra_lb_scheduler, make_dyntra_test_config
from tests.ut.kv_offload.utils import create_model_runner_output, create_request
from vllm_ascend.patch.worker import v41_prefill_policy as policy
from vllm_ascend.patch.worker.v41_decode_first import decode_first_config


def make_scheduler(asynchronous=False, policy_name="fcfs", budget=8192, enabled=True):
    config = make_dyntra_test_config(max_num_seqs=64, max_num_batched_tokens=budget, block_size=16)
    config.kv_transfer_config = None
    config.model_config.max_model_len = 32768
    config.scheduler_config.policy = policy_name
    config.scheduler_config.async_scheduling = asynchronous
    config.additional_config = {"enable_dsv41_decode_first": enabled}
    return create_dyntra_lb_scheduler(config, AsyncScheduler if asynchronous else Scheduler)


def add_decode(scheduler, count=1):
    requests = [create_request(i + 1, num_tokens=16) for i in range(count)]
    for request in requests:
        scheduler.add_request(request)
    output = scheduler.schedule()
    scheduler.update_from_output(output, create_model_runner_output(requests))
    return requests


def add_prefill(scheduler, request_id, length=16000):
    request = create_request(request_id, num_tokens=length)
    scheduler.add_request(request)
    return request


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("policy_name", ["fcfs", "priority"])
def test_running_and_new_prefills_share_all_remaining_budget(asynchronous, policy_name):
    scheduler = make_scheduler(asynchronous, policy_name)
    decodes = add_decode(scheduler, 32)
    running_p = add_prefill(scheduler, 100, length=20000)
    # Admit a chunk without consuming D state, then put P first in RUNNING.
    admission = scheduler.schedule(blocked_request_ids=frozenset(r.request_id for r in decodes))
    assert admission.num_scheduled_tokens[running_p.request_id] == 8192
    scheduler.running.remove(running_p)
    scheduler.running.insert(0, running_p)
    new_p = add_prefill(scheduler, 101)
    # DSpark's target rows and per-request draft workspace must both fit.
    scheduler.vllm_config.speculative_config = SimpleNamespace(max_num_new_slots_for_drafting=5)
    scheduler.num_spec_tokens = 5
    for request in decodes:
        request.spec_token_ids = [-1] * 5
    output = scheduler.schedule()
    ids = list(output.num_scheduled_tokens)
    assert ids[:32] == [request.request_id for request in decodes]
    assert all(output.num_scheduled_tokens[request.request_id] == 6 for request in decodes)
    p_tokens = sum(output.num_scheduled_tokens.get(r.request_id, 0) for r in [running_p, new_p])
    assert p_tokens == 8192 - 32 * (6 + 5) - 5
    assert p_tokens > 4000
    assert output.total_num_scheduled_tokens + 5 * len(ids) == 8192


def test_no_decode_allows_full_capacity_and_no_hidden_prefill_cap():
    scheduler = make_scheduler()
    p = add_prefill(scheduler, 100)
    assert scheduler.schedule().num_scheduled_tokens == {p.request_id: 8192}


def test_many_prefills_share_the_remainder_across_running_and_waiting():
    scheduler = make_scheduler()
    (decode,) = add_decode(scheduler)
    first = add_prefill(scheduler, 100, 256)
    second = add_prefill(scheduler, 101, 16000)
    output = scheduler.schedule()
    assert list(output.num_scheduled_tokens) == [decode.request_id, first.request_id, second.request_id]
    assert output.num_scheduled_tokens[first.request_id] == 256
    assert output.num_scheduled_tokens[second.request_id] == 8192 - 1 - 256


@pytest.mark.parametrize("asynchronous", [False, True])
def test_waiting_resumed_decode_precedes_running_prefill(asynchronous):
    scheduler = make_scheduler(asynchronous)
    (decode,) = add_decode(scheduler)
    scheduler.running.remove(decode)
    decode.status = RequestStatus.PREEMPTED
    scheduler.kv_holding_waiting.add_request(decode)
    p = add_prefill(scheduler, 100)
    output = scheduler.schedule()
    assert list(output.num_scheduled_tokens) == [decode.request_id, p.request_id]


def test_barrier_blocks_dependent_first_decode_but_allows_other_decode():
    scheduler = make_scheduler(True)
    (decode,) = add_decode(scheduler)
    p = add_prefill(scheduler, 100, 8)
    before = policy.prefill_ids(scheduler.requests, True)
    output = scheduler.schedule()
    assert policy.tag_output(output, before)
    assert p.num_computed_tokens >= p.num_prompt_tokens
    new_p = add_prefill(scheduler, 101)
    engine = SimpleNamespace(scheduler=scheduler, async_scheduling=True, _should_throttle_prefills=lambda: False)
    queue = deque([(Mock(), output, Mock())])
    assert policy.can_schedule(scheduler, queue, True)
    following = policy.schedule(engine, queue)
    assert following.num_scheduled_tokens == {decode.request_id: 1}
    assert p.request_id not in following.num_scheduled_tokens
    assert new_p.request_id not in following.num_scheduled_tokens
    scheduler.update_from_output(output, create_model_runner_output([decode, p]))
    queue.clear()
    following = policy.schedule(engine, queue)
    assert p.request_id in following.num_scheduled_tokens
    assert new_p.request_id in following.num_scheduled_tokens


@pytest.mark.parametrize("policy_name", ["fcfs", "priority"])
def test_prefill_allocation_failure_cannot_preempt_decode_or_barrier(monkeypatch, policy_name):
    scheduler = make_scheduler(policy_name=policy_name)
    (decode,) = add_decode(scheduler)
    p = add_prefill(scheduler, 100, 16000)
    allocate = scheduler.kv_cache_manager.allocate_slots
    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "allocate_slots",
        lambda request, *a, **kw: None if request is p else allocate(request, *a, **kw),
    )
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {decode.request_id: 1}
    assert decode in scheduler.running and decode.num_preemptions == 0


def test_disabled_policy_preserves_global_barrier():
    scheduler = make_scheduler(enabled=False)
    output = SimpleNamespace(num_scheduled_tokens={"p": 8})
    policy.tag_output(output, {"p"})
    assert not policy.can_schedule(scheduler, [(None, output)], True)
    assert policy.can_schedule(scheduler, [(None, output)], False)
    assert decode_first_config(SimpleNamespace(additional_config={})) is False


def test_old_untagged_barrier_is_conservative():
    scheduler = SimpleNamespace(decode_first_enabled=True)
    assert not policy.can_schedule(scheduler, [(None, SimpleNamespace(_v41_prefill_barrier=True))], True)
