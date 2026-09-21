# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops import rope_dsv4 as rope


@pytest.fixture
def rope_state(monkeypatch):
    state = rope.RopeGlobalState()
    monkeypatch.setattr(rope, "_ROPE_STATE", state)
    return state


def initialize(state, dtype):
    torch.manual_seed(17)
    cos = torch.randn(128, 1, 1, 64, dtype=dtype, device="npu")
    sin = torch.randn_like(cos)
    state.full_rope_cache["test"] = (cos, sin)
    state.registry_summary["test"] = {"default", "compressed"}
    state.layer_info["layer"] = ("test", ["default"])
    for group in state.registry_summary["test"]:
        state.runtime_buffer.setdefault("test", {})[group] = (
            torch.full((32, 1, 1, 64), 11, dtype=dtype, device="npu"),
            torch.full((32, 1, 1, 64), 13, dtype=dtype, device="npu"),
        )
        state.spec_runtime_buffer.setdefault("test", {})[group] = (
            torch.full((3, 32, 1, 1, 64), 11, dtype=dtype, device="npu"),
            torch.full((3, 32, 1, 1, 64), 13, dtype=dtype, device="npu"),
        )
    return cos, sin


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("draft_index", [None, 1, 3])
@pytest.mark.parametrize("count", [0, 1, 8, 32])
def test_cached_rope_matches_indexing(rope_state, dtype, draft_index, count):
    cos, sin = initialize(rope_state, dtype)
    positions = (torch.arange(count, device="npu", dtype=torch.int64) * 7) % 128
    # Repeated and nonmonotonic indices must preserve request order.
    positions = positions.flip(0)
    actual = rope.get_cos_and_sin_dsa(positions, use_cache=True, draft_index=draft_index)
    buffers = (
        rope_state.runtime_buffer["test"]["default"]
        if draft_index is None
        else tuple(x[draft_index - 1] for x in rope_state.spec_runtime_buffer["test"]["default"])
    )
    for proxy, full, buf, sentinel in zip(actual, (cos, sin), buffers, (11, 13)):
        torch.testing.assert_close(proxy["layer"], full[positions], rtol=0, atol=0)
        if count:
            assert proxy["layer"].data_ptr() == buf.data_ptr()
        assert torch.all(buf[count:] == sentinel)


def test_group_and_uncached_paths(rope_state):
    cos, sin = initialize(rope_state, torch.bfloat16)
    positions = torch.tensor([127, 0, 7, 7], device="npu")
    for use_cache in (False, True):
        actual = rope.get_cos_and_sin_dsa({"default": positions, "ignored": positions}, use_cache=use_cache)
        for proxy, full in zip(actual, (cos, sin)):
            torch.testing.assert_close(proxy["layer"], full[positions], rtol=0, atol=0)
    del rope_state.runtime_buffer["test"]["default"]
    actual = rope.get_cos_and_sin_dsa(positions, use_cache=True)
    assert actual[0]["layer"] == {}


def test_graph_replay_reads_updated_positions(rope_state):
    cos, sin = initialize(rope_state, torch.bfloat16)
    positions = torch.tensor([1, 2, 3, 4], device="npu")
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            rope.get_cos_and_sin_dsa(positions, use_cache=True)
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        actual = rope.get_cos_and_sin_dsa(positions, use_cache=True)
    positions.copy_(torch.tensor([127, 19, 19, 0], device="npu"))
    graph.replay()
    torch.npu.synchronize()
    for proxy, full in zip(actual, (cos, sin)):
        torch.testing.assert_close(proxy["layer"], full[positions], rtol=0, atol=0)
