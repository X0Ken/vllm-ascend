# SPDX-License-Identifier: Apache-2.0
"""Exercise the real RoPE proxy ABI, ownership, and metadata restoration on CPU."""

import unittest
from dataclasses import field, make_dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.forward_context import ForwardContext, override_forward_context

from vllm_ascend.models.deepseek_v41 import consumer_compaction_runtime as runtime
from vllm_ascend.ops.rope_dsv4 import _ROPE_STATE, RopeDataProxy

FIELDS = [
    "query_start_loc",
    "query_start_loc_cpu",
    "positions",
    "num_actual_tokens",
    "num_input_tokens",
    "num_prefill_tokens",
    "num_decode_tokens",
    "num_decodes",
    "max_query_len",
    "slot_mapping",
    "flat_slot_mapping",
    "cos",
    "sin",
    "ori_topk_length",
    "consumer_query_rows",
    "seq_lens",
    "num_reqs",
    "max_seq_len",
    "cache_seq_lens",
    "cmp_residual",
    "max_cache_seq_len",
    "smla_metadata",
]
Metadata = make_dataclass("Metadata", [(name, object, field(default=None)) for name in FIELDS])


def check_contract(lengths):
    ends = []
    total = 0
    for length in lengths:
        total += length
        ends.append(total)
    indices = torch.tensor([index for end in ends for index in range(end - 128, end)])
    starts = torch.arange(len(lengths) + 1, dtype=torch.int32) * 128
    positions = torch.cat([torch.arange(length) for length in lengths])
    cos = torch.arange(total * 4).reshape(total, 1, 1, 4).float()
    sin = -cos
    proxy = RopeDataProxy({"contract_config": {"query": (cos, sin)}})
    proxies = (proxy, RopeDataProxy(proxy._data, is_cos=False))
    originals = {
        name: Metadata(
            positions=positions,
            slot_mapping=torch.arange(total * 2).reshape(total, 2),
            flat_slot_mapping=torch.arange(total),
            cos=proxies[0],
            sin=proxies[1],
            num_reqs=len(lengths),
            seq_lens=torch.tensor(lengths, dtype=torch.int32),
            cache_seq_lens=torch.tensor(lengths, dtype=torch.int32),
            max_seq_len=max(lengths),
            max_cache_seq_len=max(lengths),
        )
        for name in ["swa", "long", "index"]
    }
    impl = SimpleNamespace(
        role=SimpleNamespace(is_kv_source=False, is_index_source=False, compress_ratio=1),
        swa_prefix="swa",
        long_kv_source_prefix="long",
        index_k_source_prefix="index",
        topology=SimpleNamespace(index_topk=512),
        _get_layer_metadata=lambda metadata: SimpleNamespace(swa=metadata["swa"], attention=metadata["long"]),
    )
    attn = SimpleNamespace(
        v41_impl=impl, rotary_emb=SimpleNamespace(layername="terminal_contract"), n_local_heads=8, head_dim=512
    )
    context = ForwardContext(no_compile_layers={"terminal": attn}, attn_metadata=dict(originals), slot_mapping={})
    hidden = torch.randn(total, 4, 8)
    mix = torch.randn(total, 4)
    calls = []

    def build(*args, **kwargs):
        calls.append(kwargs)
        return torch.zeros(32, dtype=torch.int32)

    with (
        patch.dict(_ROPE_STATE.layer_info, {"terminal_contract": ("contract_config", ["query"])}),
        override_forward_context(context),
        patch.object(torch.ops._C_ascend, "npu_sparse_flash_mla_metadata", build, create=True),
    ):
        output = runtime.prepare_terminal_consumer(hidden, mix, positions, indices, starts, 128, "terminal")
        assert (
            torch.equal(output[0], hidden[indices])
            and torch.equal(output[1], mix[indices])
            and torch.equal(output[2], positions[indices])
        )
        assert torch.equal(context.attn_metadata["swa"].cos["terminal_contract"], cos[indices])
        assert torch.equal(context.attn_metadata["swa"].ori_topk_length[:, 0], torch.arange(len(indices)) % 128 + 1)
        assert len(calls) == 1 and calls[0]["batch_size"] == len(lengths)
        assert originals["swa"].cos is proxies[0] and torch.equal(proxies[0]["terminal_contract"], cos)
        restored = runtime.restore_terminal_consumer(output[0])
        assert torch.equal(restored, output[0]) and all(
            context.attn_metadata[name] is original for name, original in originals.items()
        )


class TestConsumerMetadataContract(unittest.TestCase):
    def test_single_prompt_proxy_and_restore(self):
        check_contract([512])

    def test_two_prompt_suffixes_keep_metadata_private(self):
        check_contract([512, 1024])


if __name__ == "__main__":
    unittest.main()
