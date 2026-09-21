# SPDX-License-Identifier: Apache-2.0

from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from vllm_ascend.utils import adapt_patch, register_ascend_customop

# Standalone /tmp pytest runs do not inherit tests/ut/conftest.py.
adapt_patch()
adapt_patch(True)
register_ascend_customop()

from vllm.model_executor.layers.linear import ColumnParallelLinear  # noqa: E402

import vllm_ascend.models.deepseek_v4_dspark as dspark  # noqa: E402
from vllm_ascend.ops.linear_op import SequenceColumnParallelOp, _get_column_parallel_op  # noqa: E402


@pytest.mark.parametrize(
    ("enabled", "sp", "sp_bypass", "tp", "hidden", "uses_column"),
    [
        (False, False, False, 8, 4096, False),
        (True, False, False, 8, 4096, True),
        (True, False, False, 2, 4096, True),
        (True, True, False, 8, 4096, False),
        (True, False, True, 8, 4096, False),
        (True, False, False, 1, 4096, False),
        (True, False, False, 3, 4096, False),
    ],
)
def test_main_projection_opt_in_and_sp_fallback(enabled, sp, sp_bypass, tp, hidden, uses_column):
    config = SimpleNamespace(hidden_size=hidden, dspark_target_layer_ids=[3, 17, 31])
    with (
        patch.object(dspark, "get_ascend_config", return_value=SimpleNamespace(dspark_main_proj_tp=enabled)),
        patch.object(dspark, "enable_sp", return_value=sp),
        patch.object(dspark, "enable_sp_by_pass", return_value=sp_bypass),
        patch.object(dspark, "get_tensor_model_parallel_world_size", return_value=tp),
        patch.object(dspark, "ColumnParallelLinear") as column,
        patch.object(dspark, "ReplicatedLinear") as replicated,
    ):
        result = dspark._build_dspark_main_proj(config, "model.layers.61.main_proj")
    selected, other = (column, replicated) if uses_column else (replicated, column)
    other.assert_not_called()
    assert result is selected.return_value
    assert selected.call_args.args == (3 * hidden, hidden)
    assert selected.call_args.kwargs == {
        "bias": False,
        "return_bias": False,
        "quant_config": None,
        "prefix": "model.layers.61.main_proj",
        **({"gather_output": True} if uses_column else {}),
    }


def test_main_projection_prefix_is_classified_as_sequence_input_when_sp_enabled():
    with (
        patch("vllm_ascend.ops.linear_op.enable_dsa_cp", return_value=False),
        patch("vllm_ascend.ops.linear_op.enable_sp", return_value=True),
    ):
        op = _get_column_parallel_op("model.layers.61.main_proj", MagicMock())
    # This is why the opt-in projection must retain its local SP fallback.
    assert isinstance(op, SequenceColumnParallelOp)


@pytest.mark.parametrize("tp", [1, 2, 8])
def test_checkpoint_main_projection_weight_is_sharded_once_and_in_rank_order(tp):
    hidden, input_size = 16, 48
    full = torch.arange(hidden * input_size, dtype=torch.float32).reshape(hidden, input_size)
    shards = []
    for rank in range(tp):
        model = dspark.DSparkDeepseekV4ForCausalLM.__new__(dspark.DSparkDeepseekV4ForCausalLM)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(num_attention_heads=8, num_hidden_layers=61)
        model.model = nn.Module()
        model.model.num_dspark_layers = 3
        model.model.get_expert_mapping = lambda: []
        projection = nn.Module()
        projection.weight = nn.Parameter(torch.empty(hidden // tp, input_size), requires_grad=False)
        projection.weight.output_dim = 0
        projection.register_parameter("bias", None)
        loader = SimpleNamespace(tp_rank=rank)
        projection.weight.weight_loader = partial(ColumnParallelLinear.weight_loader, loader)
        layer = nn.Module()
        layer.main_proj = projection
        model.model.layers = nn.ModuleDict({"61": layer})
        model.model.main_proj = projection  # Same alias order as the real model.
        with (
            patch.object(dspark, "get_tensor_model_parallel_world_size", return_value=tp),
            patch.object(dspark, "get_tensor_model_parallel_rank", return_value=rank),
        ):
            loaded = model.load_weights([("mtp.0.main_proj.weight", full)])
        assert loaded == {"model.layers.61.main_proj.weight"}
        torch.testing.assert_close(projection.weight, full.chunk(tp, dim=0)[rank])
        assert projection.bias is None
        shards.append(projection.weight.detach())
    torch.testing.assert_close(torch.cat(shards, dim=0), full)
