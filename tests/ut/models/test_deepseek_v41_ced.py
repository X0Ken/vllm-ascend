# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec
from vllm_ascend.models.deepseek_v41.ced import (
    get_ced_role,
    is_producer_weight,
    received_replay_positions,
    received_state_replay_skip,
    replay_cache_origin,
)
from vllm_ascend.models.deepseek_v41.dspark import DeepseekV41DSparkSWACache
from vllm_ascend.models.deepseek_v41.model import AscendDeepseekV41SWACache


def configuration(role="off", **extra):
    return SimpleNamespace(
        additional_config={"dsv41_ced_role": role},
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            data_parallel_size=1,
        ),
        speculative_config=None,
        cache_config=SimpleNamespace(swa_bounded_replay=True),
        kv_transfer_config=SimpleNamespace(
            kv_connector="MooncakeHybridConnector",
            kv_role="kv_producer" if role == "prefill" else "kv_consumer",
            kv_connector_extra_config={"named_cache_transfer": True},
        ),
        **extra,
    )


def test_default_off_needs_no_pd_resources():
    assert get_ced_role(SimpleNamespace(additional_config={})) == "off"


@pytest.mark.parametrize("role", [True, "producer", "on", None])
def test_unknown_roles_fail(role):
    with pytest.raises(ValueError):
        get_ced_role(configuration(role))


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_explicit_roles_require_checked_transfer(role):
    config = configuration(role)
    assert get_ced_role(config) == role
    config.kv_transfer_config.kv_connector_extra_config = {}
    with pytest.raises(ValueError, match="named_cache_transfer"):
        get_ced_role(config)


def test_producer_rejects_speculation_and_unbounded_replay():
    config = configuration("prefill")
    config.speculative_config = object()
    with pytest.raises(ValueError, match="speculative"):
        get_ced_role(config)
    config.speculative_config = None
    config.cache_config.swa_bounded_replay = False
    with pytest.raises(ValueError, match="bounded replay"):
        get_ced_role(config)


@pytest.mark.parametrize(
    "axis",
    ["pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size", "data_parallel_size"],
)
def test_unsupported_topologies_fail(axis):
    config = configuration("prefill")
    setattr(config.parallel_config, axis, 2)
    with pytest.raises(ValueError, match="DP1"):
        get_ced_role(config)


def test_weight_filter_retains_boundary_and_drops_upper_decoder():
    assert is_producer_weight("model.layers.19.mlp.experts.0.gate_proj.weight")
    assert is_producer_weight("layers.20.attn.wkv.weight")
    assert is_producer_weight("model.embed_tokens.weight")
    assert not is_producer_weight("model.layers.21.mlp.experts.0.gate_proj.weight")
    assert not is_producer_weight("model.layers.39.attn.wkv.weight")


def test_mixed_replay_new_tail_and_unreceived_state():
    # 128 replay rows; replay+one new row; a later replay chunk; an ordinary
    # local hit without received private state; decode beyond the replay.
    start = torch.tensor([1024, 1024, 1024, 1024, 1024])
    length = torch.tensor([128, 129, 64, 128, 1])
    seq = torch.tensor([1152, 1153, 1152, 1152, 1154])
    received = torch.tensor([True, True, True, False, True])
    assert received_state_replay_skip(start, seq, length, length, received).tolist() == [128, 128, 64, 0, 0]


def test_replay_mask_excludes_boundaries_and_unreceived_requests():
    positions = torch.tensor([1023, 1024, 1151, 1152, 1024, 5])
    start = torch.tensor([1024, 1024, 1024, 1024, 1024, 0])
    received = torch.tensor([True, True, True, True, False, True])
    assert received_replay_positions(positions, start, received).tolist() == [False, True, True, False, False, False]


def test_replay_origin_preserves_decode_history_in_mixed_batches():
    start = torch.tensor([1024, 1024, 0, 0])
    seq = torch.tensor([1153, 1124, 10000, 64])
    query = torch.tensor([129, 20, 1, 64])
    assert replay_cache_origin(start, seq, query).tolist() == [1024, 1024, 9872, 0]


@pytest.mark.parametrize("role, private", [("off", False), ("decode", True)])
def test_ced_draft_swa_is_private_until_rebuilt(monkeypatch, role, private):
    spec = AscendSlidingWindowMLASpec(
        block_size=128,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=128,
        model_version="deepseek_v41",
        bounded_replay=False,
    )
    monkeypatch.setattr(AscendDeepseekV41SWACache, "get_kv_cache_spec", lambda self, config: spec)
    cache = DeepseekV41DSparkSWACache.__new__(DeepseekV41DSparkSWACache)
    actual = cache.get_kv_cache_spec(SimpleNamespace(additional_config={"dsv41_ced_role": role}))
    assert actual.bounded_replay == private
    assert actual.prefix_cacheable != private
