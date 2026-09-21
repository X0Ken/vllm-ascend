# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import moe_comm_method
from vllm_ascend.ops.fused_moe.moe_mlp import cumsum_group_list
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEQuantParams,
    MoERoutingParams,
    build_fused_experts_input,
    build_mlp_compute_input,
    build_token_dispatch_input,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import AscendDeviceType


def _input():
    return build_fused_experts_input(
        hidden_states=torch.ones((2, 16), dtype=torch.bfloat16),
        topk_weights=torch.ones((2, 2)),
        topk_ids=torch.tensor([[0, 2], [3, 2]], dtype=torch.int32),
        w1=torch.empty(0),
        w2=torch.empty(0),
        quant_type=QuantType.W8A8,
        dynamic_eplb=False,
    )


@pytest.mark.parametrize(
    "case",
    ["eligible", "fusion_off", "eplb", "w4a8", "bf16", "activation", "offset", "scale_bias", "lora", "mc2", "a5"],
)
def test_cumulative_routing_eligibility(case):
    inputs = _input()
    fusion = case != "fusion_off"
    comm_type = MoECommType.MC2 if case == "mc2" else MoECommType.ALLGATHER
    device = AscendDeviceType.A5 if case == "a5" else AscendDeviceType.A2
    if case == "eplb":
        inputs = replace(inputs, dynamic_eplb=True)
    elif case in ("w4a8", "bf16"):
        inputs = replace(inputs, quant=MoEQuantParams(quant_type=QuantType.W4A8 if case == "w4a8" else QuantType.NONE))
    elif case == "activation":
        inputs = replace(inputs, activation="swigluoai_uninterleave")
    elif case in ("offset", "scale_bias"):
        field = "w1_offset" if case == "offset" else "w1_scale_bias"
        inputs = replace(inputs, weights=replace(inputs.weights, **{field: torch.empty(0)}))
    elif case == "lora":
        inputs = replace(inputs, lora_context=object())

    with (
        patch.object(moe_comm_method, "_EXTRA_CTX", SimpleNamespace(moe_comm_type=comm_type)),
        patch.object(moe_comm_method, "get_ascend_device_type", return_value=device),
    ):
        assert moe_comm_method._use_allgather_cumulative_expert_tokens(inputs, fusion) == (case == "eligible")


@pytest.mark.parametrize("cumulative", [False, True])
@pytest.mark.parametrize("expert_offset", [0, 4])
@pytest.mark.parametrize("empty", [False, True])
def test_allgather_cumulative_contract(cumulative, expert_offset, empty):
    inputs = _input()
    counts = torch.tensor([0, 0, 0, 0] if empty else [1, 0, 2, 1], dtype=torch.int32)
    group_list = counts.cumsum(0).to(torch.int32) if cumulative else counts
    scale = torch.ones(4)
    expanded = torch.arange(4, dtype=torch.int32)
    sorted_hidden = torch.zeros((4, 16), dtype=torch.int8)
    expert_map = torch.full((8,), -1, dtype=torch.int32)
    expert_map[expert_offset : expert_offset + 4] = torch.arange(4, dtype=torch.int32)
    inputs = replace(
        inputs,
        topk_ids=inputs.topk_ids + expert_offset,
        routing=MoERoutingParams(
            expert_map=expert_map,
            global_redundant_expert_num=0,
            mc2_mask=None,
            apply_router_weight_on_input=False,
        ),
    )
    dispatch_input = build_token_dispatch_input(fused_experts_input=inputs, cumulative_expert_tokens=cumulative)
    dispatcher = TokenDispatcherWithAllGather(top_k=2, num_experts=8, num_local_experts=4)
    with (
        patch(
            "vllm_ascend.ops.fused_moe.token_dispatcher.get_ep_group",
            return_value=SimpleNamespace(rank_in_group=expert_offset // 4),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.token_dispatcher.DeviceOperator.npu_moe_init_routing",
            return_value=(sorted_hidden, expanded, group_list, scale),
        ) as routing,
    ):
        result = dispatcher.token_dispatch(dispatch_input)
    expected_type = 0 if cumulative else 1
    assert routing.call_args.kwargs["expert_tokens_num_type"] == expected_type
    assert routing.call_args.kwargs["active_expert_range"] == [expert_offset, expert_offset + 4]
    assert routing.call_args.kwargs["quant_mode"] == 1
    assert result.group_list_type == expected_type
    assert result.group_list.dtype == torch.int64
    torch.testing.assert_close(result.group_list, group_list.to(torch.int64))
    assert result.dynamic_scale is scale
    assert result.combine_metadata.expanded_row_idx is expanded
    # Both GMMs receive the representation tag; combine uses row indices and
    # router weights and is independent of the group-list representation.
    mlp = build_mlp_compute_input(fused_experts_input=inputs, token_dispatch_output=result, use_fusion_ops=True)
    assert mlp.group_list is result.group_list
    assert mlp.group_list_type == expected_type
    assert mlp.fusion
    prefix = cumsum_group_list(mlp.group_list, mlp.group_list_type, 0)
    torch.testing.assert_close(prefix, counts.to(torch.int64).cumsum(0))
    if cumulative:
        assert prefix is mlp.group_list


def test_dispatch_input_defaults_to_counts():
    assert not build_token_dispatch_input(fused_experts_input=_input()).cumulative_expert_tokens
