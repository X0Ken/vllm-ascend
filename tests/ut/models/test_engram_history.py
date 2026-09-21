# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only regression tests for external KV Engram history restoration."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "engram_hash", Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/engram_hash.py"
)
hash_mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hash_mod
spec.loader.exec_module(hash_mod)


@pytest.mark.parametrize("query_tokens", [1, 20])
@pytest.mark.parametrize("stale", [False, True])
@pytest.mark.parametrize("barrier", [None, 98, 99])
def test_external_kv_rebuilds_engram_prompt_tail(query_tokens, stale, barrier):
    def history():
        h = hash_mod.PagedNgramHistory.__new__(hash_mod.PagedNgramHistory)
        h.token_map = torch.arange(100)
        h.pad_id = 2
        h.image_token_id = 99
        h.image_pad_token_id = 98
        h.lookback = 4
        h.primes = torch.tensor([[[101, 103], [107, 109], [113, 127]]])
        h.offsets = torch.tensor([[0, 101, 204, 311, 420, 533]])
        h.multipliers = torch.tensor([[3, 5, 7, 11]])
        h.pages = {}
        return h

    prompt = [i % 80 + 3 for i in range(48)]
    if barrier is not None:
        prompt[14] = barrier
    prompt_ids = torch.tensor(prompt)
    golden = history()
    full, mask = golden.update(
        prompt_ids, torch.arange(48), torch.zeros(48, dtype=torch.long), torch.tensor([[1, 2, 3, 4, 5, 6]]), 8
    )
    resumed = history()
    new_pages = torch.tensor([[101, 102, 103, 104, 105, 106]])
    if stale:
        resumed.pages = {i: torch.full((8,), 77) for i in range(101, 107)}
    actual, actual_mask = resumed.update(
        prompt_ids[16 : 16 + query_tokens],
        torch.arange(16, 16 + query_tokens),
        torch.zeros(query_tokens, dtype=torch.long),
        new_pages,
        8,
        prompt_token_ids={0: prompt},
    )
    assert torch.equal(actual, full[16 : 16 + query_tokens])
    assert torch.equal(actual_mask, mask[16 : 16 + query_tokens])
    # Only the lookback tail is restored, not the entire cached prefix.
    if not stale:
        assert 101 not in resumed.pages
        assert torch.equal(resumed.pages[102][:5], torch.full((5,), -1))
