# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional INT8 Engram lookup from registered CPU memory.

Adapted from upstream 0deca3181. Registrations live for the model lifetime;
close() is for explicit teardown after all device users have finished.
"""

import ctypes
from functools import cache

import torch
from vllm.triton_utils import tl, triton

SCALE_GROUP = 32
CHUNK_ROWS = 1 << 22
ACL_HOST_REG_MAPPED = 0x2
ACL_HOST_REG_PINNED = 0x10000000


@cache
def _host_library() -> ctypes.CDLL:
    """The CANN runtime entry points that publish host memory to the device."""

    lib = ctypes.CDLL("libascendcl.so")
    lib.aclrtMallocHost.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint32]
    lib.aclrtMallocHost.restype = ctypes.c_int
    lib.aclrtFreeHost.argtypes = [ctypes.c_void_p]
    lib.aclrtFreeHost.restype = ctypes.c_int
    lib.aclrtHostRegisterV2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32]
    lib.aclrtHostRegisterV2.restype = ctypes.c_int
    lib.aclrtHostGetDevicePointer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32]
    lib.aclrtHostGetDevicePointer.restype = ctypes.c_int
    lib.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
    lib.aclrtHostUnregister.restype = ctypes.c_int
    return lib


class HostUvaBuffer:
    """Host memory the device gathers from directly."""

    def __init__(self, shape, dtype, device):
        self.lib = _host_library()
        self.device = torch.device(device)
        self.registered = False
        rows = int(shape[0])
        row_elements = int(torch.Size(shape[1:]).numel())
        self.row_bytes = row_elements * torch.empty((), dtype=dtype).element_size()
        size = rows * self.row_bytes
        if size <= 0:
            raise ValueError("UVA buffers must be nonempty")
        self.pointer = ctypes.c_void_p()
        rc = self.lib.aclrtMallocHost(ctypes.byref(self.pointer), size, 0)
        if rc:
            raise RuntimeError(f"aclrtMallocHost failed: rc={rc} size={size}")
        self.buffer = (ctypes.c_char * size).from_address(self.pointer.value)
        self.tensor = torch.frombuffer(self.buffer, dtype=dtype).reshape(shape)
        rc = self.lib.aclrtHostRegisterV2(self.pointer, size, ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED)
        if rc:
            self.tensor = None
            self.buffer = None
            self.lib.aclrtFreeHost(self.pointer)
            self.pointer = ctypes.c_void_p()
            raise RuntimeError(f"aclrtHostRegisterV2 failed: rc={rc} size={size}")
        self.registered = True
        address = ctypes.c_void_p()
        rc = self.lib.aclrtHostGetDevicePointer(self.pointer, ctypes.byref(address), 0)
        if rc:
            self.close()
            raise RuntimeError(f"aclrtHostGetDevicePointer failed: rc={rc}")
        try:
            self.ptrs = torch.tensor(
                [address.value + start * self.row_bytes for start in range(0, rows, CHUNK_ROWS)],
                dtype=torch.int64,
                device=device,
            )
        except BaseException:
            self.close()
            raise

    def close(self):
        """Explicit, idempotent teardown; callers must release tensor aliases."""
        if not self.pointer.value:
            return
        torch.npu.synchronize(self.device)
        if self.registered:
            rc = self.lib.aclrtHostUnregister(self.pointer)
            if rc:
                raise RuntimeError(f"aclrtHostUnregister failed: rc={rc}")
            self.registered = False
        self.tensor = None
        self.buffer = None
        rc = self.lib.aclrtFreeHost(self.pointer)
        if rc:
            raise RuntimeError(f"aclrtFreeHost failed: rc={rc}")
        self.pointer = ctypes.c_void_p()


@triton.jit
def _engram_host_uva_gather_dequant_kernel(
    codes_ptrs,
    scales_ptrs,
    ids,
    output,
    rows,
    CHUNK: tl.constexpr,
    WIDTH: tl.constexpr,
    GROUP: tl.constexpr,
):
    row = tl.program_id(0)
    if row < rows:
        index = tl.load(ids + row).to(tl.int64)
        chunk = index // CHUNK
        local = index % CHUNK
        codes = tl.load(codes_ptrs + chunk).to(tl.pointer_type(tl.int8))
        scales = tl.load(scales_ptrs + chunk).to(tl.pointer_type(tl.float32))
        col = tl.arange(0, WIDTH)
        value = tl.load(codes + local * WIDTH + col).to(tl.float32)
        scale = tl.load(scales + local * (WIDTH // GROUP) + col // GROUP)
        tl.store(output + row * WIDTH + col, (value * scale).to(tl.bfloat16))


def gather_dequantize_host_uva(codes: HostUvaBuffer, scales: HostUvaBuffer, ids: torch.Tensor) -> torch.Tensor:
    """Gather rows ``ids`` from a registered host table and dequantize on device."""

    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    width = codes.tensor.shape[-1]
    rows = ids.numel()
    output = torch.empty((rows, width), dtype=torch.bfloat16, device=ids.device)
    if rows == 0:
        return output
    init_device_properties_triton()
    _engram_host_uva_gather_dequant_kernel[(rows,)](
        codes.ptrs,
        scales.ptrs,
        ids.reshape(-1).to(torch.int64),
        output,
        rows,
        CHUNK=CHUNK_ROWS,
        WIDTH=width,
        GROUP=SCALE_GROUP,
        num_warps=4,
    )
    return output
