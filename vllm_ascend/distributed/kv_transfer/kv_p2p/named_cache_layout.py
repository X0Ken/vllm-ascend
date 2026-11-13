# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checked, name-based transfers between different hybrid cache layouts.

An address in a handshake is a *view*, not necessarily a contiguous page.
Keep logical tensor planes separate from physical registration regions. This
also prevents P-side layer pruning from shifting D-side cache group indices.
The protocol is experimental and must be enabled on both peers.
"""

from dataclasses import asdict, dataclass
from math import prod


@dataclass(frozen=True)
class CachePlane:
    name: str
    plane: int
    group: int
    base: int
    blocks: int
    stride: int
    shape: tuple[int, ...]
    dtype: str
    segments: tuple[tuple[int, int], ...]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "CachePlane":
        value = dict(value)
        value["shape"] = tuple(value["shape"])
        value["segments"] = tuple(tuple(segment) for segment in value["segments"])
        return cls(**value)


def describe_plane(name, plane, group, tensor, num_blocks: int) -> CachePlane:
    """Describe a block-major view without reading any device values."""
    shape = tuple(tensor.shape)
    strides = tuple(tensor.stride())
    itemsize = tensor.element_size()
    if not shape or shape[0] != num_blocks or any(s <= 0 for s in strides):
        raise ValueError(f"Unsupported block-major KV view: {name}, {shape}, {strides}")
    # Collapse the contiguous suffix; enumerate only genuinely strided axes.
    suffix = len(shape)
    contiguous = 1
    while suffix > 1 and strides[suffix - 1] == contiguous:
        suffix -= 1
        contiguous *= shape[suffix]
    offsets = [0]
    for size, stride in zip(shape[1:suffix], strides[1:suffix]):
        offsets = [offset + index * stride for offset in offsets for index in range(size)]
    segments = tuple((offset * itemsize, contiguous * itemsize) for offset in offsets)
    if sum(length for _, length in segments) != prod(shape[1:]) * itemsize:
        raise ValueError(f"Incomplete KV plane: {name}")
    if any(offset + length > strides[0] * itemsize for offset, length in segments):
        raise ValueError(f"KV plane crosses block stride: {name}")
    return CachePlane(
        name,
        plane,
        group,
        tensor.data_ptr(),
        num_blocks,
        strides[0] * itemsize,
        shape[1:],
        str(tensor.dtype),
        segments,
    )


def plan_named_transfer(
    local,
    remote,
    local_blocks,
    remote_blocks,
    *,
    optional_names=frozenset(),
    local_offsets=None,
    remote_offsets=None,
):
    """Return local destinations, remote sources, and byte lengths.

    Groups are resolved independently on each peer. Missing required planes,
    mismatched logical layouts, and invalid block IDs fail before any DMA.
    Python integers deliberately preserve address arithmetic above 4 GiB.
    """
    remote_by_name = {(plane.name, plane.plane): plane for plane in remote}
    if len(remote_by_name) != len(remote):
        raise ValueError("Duplicate remote KV plane")
    destinations, sources, lengths = [], [], []
    for dst in local:
        src = remote_by_name.get((dst.name, dst.plane))
        if src is None:
            if dst.name in optional_names:
                continue
            raise ValueError(f"Missing remote KV plane: {dst.name}:{dst.plane}")
        if (dst.shape, dst.dtype) != (src.shape, src.dtype):
            raise ValueError(f"Incompatible remote KV plane: {dst.name}")
        if tuple(n for _, n in dst.segments) != tuple(n for _, n in src.segments):
            raise ValueError(f"Incompatible KV plane segments: {dst.name}")
        dst_ids, src_ids = local_blocks[dst.group], remote_blocks[src.group]
        if not dst_ids:
            continue
        dst_start = 0 if local_offsets is None else local_offsets[dst.group]
        src_start = 0 if remote_offsets is None else remote_offsets[src.group]
        offset = dst_start - src_start
        if offset < 0 or offset + len(dst_ids) > len(src_ids):
            raise ValueError(f"Remote KV block list is too short: {dst.name}")
        src_ids = src_ids[offset : offset + len(dst_ids)]
        for dst_id, src_id in zip(dst_ids, src_ids):
            if not (0 < dst_id < dst.blocks and 0 < src_id < src.blocks):
                raise ValueError(f"Invalid/null KV block ID: {dst.name}, {dst_id}, {src_id}")
            for (dst_offset, length), (src_offset, _) in zip(dst.segments, src.segments):
                destinations.append(dst.base + dst_id * dst.stride + dst_offset)
                sources.append(src.base + src_id * src.stride + src_offset)
                lengths.append(length)
    return destinations, sources, lengths
