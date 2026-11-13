# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise DP admission consensus and exact dependency policy without an NPU."""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
source = ROOT / "vllm_ascend/models/deepseek_v41/prefill_compaction.py"
node = next(
    n
    for n in ast.parse(source.read_text()).body
    if isinstance(n, ast.FunctionDef) and n.name == "coordinated_terminal_count"
)
scope = {}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
plan = scope["coordinated_terminal_count"]


@pytest.mark.parametrize(
    "eligible,benefit,counts,capacities,expected",
    [
        ([1, 1], [1, 0], [1, 96], [8192, 8192], 96),
        ([1, 1], [1, 0], [0, 0], [8192, 8192], 8),
        ([1, 1], [0, 0], [96, 96], [192, 192], 0),
        ([1, 0], [1, 0], [1, 96], [8192, 8192], 0),
        ([1, 1], [1, 0], [1, 96], [8192, 32], 0),
        ([1, 1], [1, 1], [5, 3], [512, 512], 8),
    ],
)
def test_dp_consensus_including_mixed_decode_idle_and_scoring(eligible, benefit, counts, capacities, expected):
    assert plan(eligible, benefit, counts, capacities) == expected


def test_real_k5_feature_boundary_and_prefix_reuse_block_multilayer_trimming():
    path = ROOT / "vllm_ascend/models/deepseek_v41/consumer_compaction.py"
    spec = importlib.util.spec_from_file_location("cut_policy", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    config = dict(
        num_hidden_layers=40,
        sliding_window=128,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        engram_layer_ids=[1, 14],
        dspark_target_layer_ids=[37, 38, 39],
        candidate_source_layer_id=20,
    )
    p = m.plan_consumer_compaction(config, 5120, requires_full_hc=False, compact_metadata_supported=True)
    assert (p.boundary, p.remaining_layers, p.retained_tokens) == (39, 1, 128)
    assert m.terminal_suffix_layout([0, 5120], 128, requires_reusable_cache=True) is None


def test_multimodal_and_direct_model_ownership():
    from types import SimpleNamespace

    node = next(
        n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "text_backbone"
    )
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    backbone = object()
    direct = SimpleNamespace(model=backbone)
    wrapped = SimpleNamespace(language_model=direct)
    assert scope["text_backbone"](direct) is backbone
    assert scope["text_backbone"](wrapped) is backbone


def test_sp_dense_decode_or_idle_peer_keeps_its_own_collective_length():
    assert plan([1, 1], [1, 0], [1, 96], [1024, 12], allow_dense_peers=True) == 96
    assert plan([1, 1], [1, 0], [1, 0], [640, 1], allow_dense_peers=True) == 8
    assert plan([1, 1], [1, 0], [1, 96], [12, 12], allow_dense_peers=True) == 0
