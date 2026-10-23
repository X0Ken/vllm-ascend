# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import sys


def test_lookback_boundary_with_expandable_segments():
    # Start a fresh process so the allocator is configured before NPU setup.
    result = subprocess.run(
        [sys.executable, __file__, "--boundary-worker"],
        env={**os.environ, "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "96 boundary cases passed" in result.stdout


def _run_boundary_cases():
    # Imports stay in the worker to keep the parent allocator uninitialized.
    import torch
    import torch_npu  # noqa: F401

    from vllm_ascend.ops.triton.engram_lookback import _gather_lookback_kernel

    rows, stride, depth = 32, 1048576, 3
    device = "npu:0"
    # Slot zero starts at the allocation boundary. A negative history offset
    # must be safe even when the load is masked out, including padded rows.
    token_ids = torch.arange(rows * stride, dtype=torch.int32, device=device).view(rows, stride)
    cases = 0
    for history in [0, 1, 2, 3, 27, stride]:
        for count in [0, 1, 2, 4, 8, 16, 31, rows]:
            for reverse in [False, True]:
                mapping = list(range(rows - 1, rows - count - 1, -1)) if reverse else list(range(count))
                # Keep a valid index allocation even for an empty batch.
                indices = torch.tensor(mapping or [0], dtype=torch.int64, device=device)
                computed = torch.full((rows,), history, dtype=torch.int32, device=device)
                output = torch.full((rows, depth), -99, dtype=torch.int32, device=device)
                _gather_lookback_kernel[(rows,)](
                    output, indices, computed, token_ids, stride, count, DEPTH=depth, BLOCK_DEPTH=4
                )
                torch.npu.synchronize()
                expected = [
                    [row * stride + history - 1 - offset if history > offset else -1 for offset in range(depth)]
                    for row in mapping
                ] + [[-1] * depth] * (rows - count)
                assert output.cpu().tolist() == expected, (history, count, reverse)
                cases += 1
    print(f"{cases} boundary cases passed", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--boundary-worker"]:
        _run_boundary_cases()
    else:
        test_lookback_boundary_with_expandable_segments()
