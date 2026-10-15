# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm_ascend.worker.mm_prefill import has_scheduled_multimodal_tokens


def request(computed, spans=()):
    return SimpleNamespace(
        num_computed_tokens=computed,
        mm_features=[
            SimpleNamespace(mm_position=SimpleNamespace(offset=offset, length=length)) for offset, length in spans
        ],
    )


@pytest.mark.parametrize(
    "computed,count,spans,expected",
    [
        (0, 204, [(2, 189)], True),  # Cached encoder output still needs image routing.
        (100, 32, [(2, 189)], True),  # Chunk begins inside an image.
        (0, 2, [(2, 189)], False),  # Exclusive right boundary.
        (191, 32, [(2, 189)], False),  # Text after an image.
        (204, 6, [(2, 189)], False),  # Speculative decode stays compiled.
        (0, 0, [(2, 189)], False),
        (0, 204, [], False),  # Text-only prefill stays compiled.
        (210, 100, [(2, 189), (300, 100)], True),
    ],
)
def test_scheduled_span_intersection(computed, count, spans, expected):
    assert has_scheduled_multimodal_tokens({"a": request(computed, spans)}, {"a": count}) is expected


def test_mixed_batch_checks_every_request():
    requests = {"text": request(0), "image": request(0, [(2, 189)])}
    assert has_scheduled_multimodal_tokens(requests, {"text": 204, "image": 204})
    assert not has_scheduled_multimodal_tokens(requests, {"text": 204})
