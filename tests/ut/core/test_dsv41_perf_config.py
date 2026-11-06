# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from vllm_ascend.patch.worker.v41_graph_config import draft_capture_sizes


def config(additional=None):
    return SimpleNamespace(
        additional_config=additional or {},
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[6, 12, 48, 96, 192]),
    )


def test_draft_default_matches_existing_capture_policy():
    c = config()
    assert draft_capture_sizes(c, 5) == [5, 6, 12, 48, 80, 96, 192]
    assert c.compilation_config.cudagraph_capture_sizes == [6, 12, 48, 96, 192]


def test_explicit_draft_sizes_do_not_follow_target_sizes():
    sizes = [160, 60, 120, 60]
    c = config({"dspark_draft_graph_capture_sizes": sizes})
    c.compilation_config.cudagraph_capture_sizes = [72, 144, 192]
    assert draft_capture_sizes(c, 5) == [5, 60, 80, 120, 160]
    assert sizes == [160, 60, 120, 60]


@pytest.mark.parametrize("sizes", [[], [0], [-1], [True], [1.5], "60", 60])
def test_invalid_draft_capture_sizes(sizes):
    with pytest.raises(ValueError):
        draft_capture_sizes(config({"dspark_draft_graph_capture_sizes": sizes}), 5)
