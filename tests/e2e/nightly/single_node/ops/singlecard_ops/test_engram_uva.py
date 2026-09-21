# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UVA lookup must preserve INT8 Engram rows across pointer chunks."""

import torch

from vllm_ascend.models.deepseek_v41.engram_uva import CHUNK_ROWS, HostUvaBuffer, gather_dequantize_host_uva


@torch.inference_mode()
def test_engram_uva_chunk_boundary_duplicates_and_empty():
    device = torch.device("npu", torch.npu.current_device())
    rows, width = CHUNK_ROWS + 17, 256
    codes = HostUvaBuffer((rows, width), torch.int8, device)
    try:
        scales = HostUvaBuffer((rows, width // 32), torch.float32, device)
        try:
            ids = torch.tensor([0, 1, 13, CHUNK_ROWS - 1, CHUNK_ROWS, rows - 1, 1])
            values = (torch.arange(ids.numel() * width).reshape(-1, width) % 251 - 125).to(torch.int8)
            values[-1] = values[1]
            sf = torch.pow(2.0, (torch.arange(ids.numel() * 8).reshape(-1, 8) % 12 - 8).float())
            sf[-1] = sf[1]
            codes.tensor[ids] = values
            scales.tensor[ids] = sf
            expected = (values.float().view(-1, 8, 32) * sf[:, :, None]).reshape(-1, width).bfloat16()
            actual = gather_dequantize_host_uva(codes, scales, ids.to(device)).cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            empty = gather_dequantize_host_uva(codes, scales, torch.empty(0, dtype=torch.int64, device=device))
            assert empty.shape == (0, width)
        finally:
            scales.close()
    finally:
        codes.close()
