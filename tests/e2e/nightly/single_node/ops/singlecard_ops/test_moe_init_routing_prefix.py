# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Standalone A2 operator equivalence check; needs torch-NPU, not vLLM.

Run: python tests/e2e/nightly/single_node/ops/singlecard_ops/test_moe_init_routing_prefix.py --device npu:0
"""

import argparse
import itertools

import torch
import torch_npu


def _routing_cases():
    # MoE routing top-k is independent of DSpark draft length. The deployed
    # model uses top-k=6; 3/5/7 remain additional operator coverage.
    cases = itertools.product([1, 4, 6, 8, 32, 192, 256], [3, 5, 6, 7], [(0, 16), (4, 8)], [False, True], [False, True])
    for rows, topk, expert_range, prequantized, no_local_tokens in cases:
        if no_local_tokens and expert_range[0] == 0:
            continue
        yield 16, 64, rows, topk, expert_range, prequantized, no_local_tokens

    # TP8 with EP disabled retains all 256 experts on each rank; the routing
    # call has expert_map=None and therefore active_expert_range=[0, 256].
    # Exercise its actual hidden width without an expensive full Cartesian
    # sweep. This is a local operator check, not a distributed TP8 test.
    for rows, prequantized in [(4, False), (6, True), (8, False), (192, True), (256, False)]:
        yield 256, 4096, rows, 6, (0, 256), prequantized, False


def run_equivalence(device: str) -> int:
    torch.npu.set_device(device)
    torch.manual_seed(20260920)
    completed = 0
    for num_experts, hidden_dim, rows, topk, (first, last), prequantized, no_local_tokens in _routing_cases():
        # Nonuniform loads, deliberately empty experts, and a nonzero local
        # expert offset. Empty-local cases are meaningful for sharded ranges.
        if no_local_tokens:
            ids = torch.arange(topk).repeat(rows, 1) + 8
        elif num_experts == 256:
            ids = (torch.arange(rows)[:, None] * 13 + torch.arange(topk)[None, :] * 37) % num_experts
        else:
            ids = (torch.arange(rows)[:, None] + torch.arange(topk)[None, :]) % 8 + 2
        ids = ids.to(device=device, dtype=torch.int32)
        hidden = torch.randn((rows, hidden_dim), device=device, dtype=torch.bfloat16)
        if prequantized:
            hidden, scale = torch_npu.npu_dynamic_quant(hidden)
            quant_mode = -1
        else:
            scale, quant_mode = None, 1
        kwargs = dict(
            scale=scale,
            active_num=rows * topk,
            expert_num=num_experts,
            expert_tokens_num_flag=True,
            active_expert_range=[first, last],
            quant_mode=quant_mode,
        )
        counts_result = torch_npu.npu_moe_init_routing_v2(hidden, ids, expert_tokens_num_type=1, **kwargs)
        prefix_result = torch_npu.npu_moe_init_routing_v2(hidden, ids, expert_tokens_num_type=0, **kwargs)
        x_count, row_count, counts, scale_count = counts_result
        x_prefix, row_prefix, prefix, scale_prefix = prefix_result
        torch.npu.synchronize()
        expected_counts = torch.bincount(ids.cpu().flatten().to(torch.int64), minlength=num_experts)[first:last]
        assert counts.dtype == prefix.dtype, (counts.dtype, prefix.dtype)
        torch.testing.assert_close(counts.cpu().to(torch.int64), expected_counts, rtol=0, atol=0)
        torch.testing.assert_close(prefix.cpu().to(torch.int64), expected_counts.cumsum(0), rtol=0, atol=0)
        valid_rows = int(expected_counts.sum())
        # Routing may reserve rows for tokens assigned to other EP ranks.
        # Only the first sum(local_counts) activation rows are consumed by GMM.
        torch.testing.assert_close(x_count[:valid_rows], x_prefix[:valid_rows], rtol=0, atol=0)
        torch.testing.assert_close(row_count, row_prefix, rtol=0, atol=0)
        assert (scale_count is None) == (scale_prefix is None)
        if scale_count is not None:
            torch.testing.assert_close(scale_count[:valid_rows], scale_prefix[:valid_rows], rtol=0, atol=0)
        if valid_rows:
            # Check the downstream GMM representation contract independently.
            weight = torch.randint(-8, 8, (last - first, hidden_dim, 32), dtype=torch.int8, device=device)
            outputs = []
            for x, group, group_type in [(x_count, counts, 1), (x_prefix, prefix, 0)]:
                outputs.append(
                    torch_npu.npu_grouped_matmul(
                        x=[x[:valid_rows].contiguous()],
                        weight=[weight],
                        group_list=group.to(torch.int64),
                        group_list_type=group_type,
                        group_type=0,
                        split_item=3,
                        output_dtype=torch.int32,
                    )[0]
                )
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        completed += 1
    torch.npu.synchronize()
    print(f"PASS: {completed} routing cases; exact counts/prefix, payload, row mapping, scales, and GMM equivalence")
    return completed


def test_moe_init_routing_prefix():
    assert run_equivalence("npu:0") == 173


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    run_equivalence(parser.parse_args().device)
