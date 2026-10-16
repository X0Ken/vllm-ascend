# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from vllm_ascend.patch.platform.patch_parallel_config import _use_sequence_parallel_moe


@pytest.mark.parametrize(
    "dp,tp,ep,expected", [(1, 8, True, False), (2, 8, True, True), (2, 1, True, False), (2, 8, False, False)]
)
def test_moe_sp_retains_p8_dp1_layout(dp, tp, ep, expected):
    config = SimpleNamespace(
        data_parallel_size=dp,
        tensor_parallel_size=tp,
        enable_expert_parallel=ep,
        all2all_backend="allgather_reducescatter",
    )
    assert _use_sequence_parallel_moe(config) is expected
