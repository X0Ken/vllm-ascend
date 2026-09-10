"""Opt-in synchronous I/O diagnostics. Never records KV contents or prompts."""

import ctypes
import hashlib
import json
import os
import threading
from pathlib import Path

ACL_MEMCPY_DEVICE_TO_HOST = 2


class KVTransferAudit:
    """Hash each actual device buffer after compute/before put, or after get.

    This intentionally adds device-to-host copies and must remain disabled in
    performance runs. Layout records map buffer indices to tensor shapes; block
    records use logical cache positions (compressed coordinates for cN groups).
    """

    def __init__(self, directory: str, tp_rank: int, pp_rank: int, pcp_rank: int, dcp_rank: int):
        import acl  # Runtime-only dependency; CPU unit tests inject the reader.

        self._memcpy = acl.rt.memcpy
        self.rank = tp_rank
        self._lock = threading.Lock()
        Path(directory).mkdir(parents=True, exist_ok=True)
        path = Path(directory) / f"tp{tp_rank}-pp{pp_rank}-pcp{pcp_rank}-dcp{dcp_rank}-{os.getpid()}.jsonl"
        self._fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)

    def _write(self, record: dict):
        record["rank"] = self.rank
        data = (json.dumps(record, separators=(",", ":")) + "\n").encode()
        with self._lock, os.fdopen(os.dup(self._fd), "ab") as output:
            output.write(data)

    def layout(self, group: int, buffers: list[dict]):
        self._write({"phase": "layout", "group": group, "buffers": buffers})

    def _checksum(self, addr: int, size: int) -> str:
        buffer = ctypes.create_string_buffer(size)
        ret = self._memcpy(ctypes.addressof(buffer), size, addr, size, ACL_MEMCPY_DEVICE_TO_HOST)
        if ret != 0:
            raise RuntimeError(f"KV audit device-to-host copy failed: {ret}")
        return hashlib.sha256(buffer.raw).hexdigest()

    def record(self, phase, request_id, group, keys, starts, ends, block_ids, addrs, sizes):
        for key, start, end, block, ptrs, lengths in zip(keys, starts, ends, block_ids, addrs, sizes, strict=True):
            self._write(
                {
                    "phase": phase,
                    "request_id": request_id,
                    "group": group,
                    "key": key,
                    "start": start,
                    "end": end,
                    "block": block,
                    "bytes": lengths,
                    "sha256": [self._checksum(ptr, length) for ptr, length in zip(ptrs, lengths, strict=True)],
                }
            )
