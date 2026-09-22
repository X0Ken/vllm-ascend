# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm_ascend.models.deepseek_v41 import model as module


def test_target_sp_preserves_full_hash_ids_and_sanitizes_placeholders(monkeypatch):
    seen = []
    hidden = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    input_ids = torch.tensor([11, -1, 12, 13, 14])

    class Layer:
        layer_idx = 0
        engram = None

        def __call__(self, positions, hidden, pre_mix, unused, input_ids):
            assert hidden.shape == (3, 2, 3)
            seen.append(input_ids.clone())
            return hidden, pre_mix

        @staticmethod
        def hc_collapse(hidden, pre_mix):
            return hidden[:, 0]

    def shard(value):
        # The selected TP rank owns three rows after padding five tokens to six.
        return value[:3]

    model = SimpleNamespace(
        use_sequence_parallel=True,
        _engram_compiled_prefetch=False,
        shared_attention_state=MagicMock(),
        hc_mult=2,
        needs_moe_input_ids=True,
        aux_hidden_state_layers=(),
        layers=[Layer()],
        norm=lambda x: x,
    )
    monkeypatch.setattr(module, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True))
    monkeypatch.setattr(module.envs, "VLLM_MOE_SKIP_PADDING", False)
    monkeypatch.setattr(module, "sp_shard", shard)
    monkeypatch.setattr(module, "sp_all_gather", lambda x: hidden)
    output = module.DeepseekV41Model.forward(
        model,
        input_ids,
        torch.arange(5),
        None,
        inputs_embeds=hidden,
        engram_lookups={},
        engram_mask=torch.ones(5, dtype=torch.bool),
    )
    torch.testing.assert_close(seen[0], torch.tensor([11, 0, 12, 13, 14]))
    torch.testing.assert_close(output, hidden)
