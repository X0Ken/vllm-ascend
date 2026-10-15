# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

import os
from pathlib import Path

import pytest

from tests.ut.distributed.ascend_store.test_backend import _make_mooncake_store_config
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import mooncake_backend as m


def test_only_checkpoint_files_are_advised_without_modification(tmp_path, monkeypatch):
    for name in ["a.safetensors", "b.safetensors", "config.json"]:
        (tmp_path / name).write_bytes(name.encode() * 100)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    calls = []

    def advise(fd, offset, length, advice):
        assert offset == length == 0 and advice == os.POSIX_FADV_DONTNEED
        assert os.read(fd, 1)
        calls.append(Path(os.readlink("/proc/self/fd/" + str(fd))).name)

    monkeypatch.setattr(m.os, "posix_fadvise", advise)
    m.MooncakeBackend._release_model_file_cache(str(tmp_path))
    assert calls == ["a.safetensors", "b.safetensors"]
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_release_closes_file_on_advice_failure(tmp_path, monkeypatch):
    (tmp_path / "one.safetensors").write_bytes(b"keep")
    fds = []

    def fail(fd, *args):
        fds.append(fd)
        raise OSError("advice failed")

    monkeypatch.setattr(m.os, "posix_fadvise", fail)
    with pytest.raises(OSError, match="advice failed"):
        m.MooncakeBackend._release_model_file_cache(str(tmp_path))
    with pytest.raises(OSError):
        os.fstat(fds[0])
    assert (tmp_path / "one.safetensors").read_bytes() == b"keep"


def test_release_requires_local_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="local safetensors"):
        m.MooncakeBackend._release_model_file_cache(str(tmp_path))


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_release_flag_is_boolean(value):
    with pytest.raises(TypeError, match="release_model_file_cache"):
        _make_mooncake_store_config(defer_setup=True, release_model_file_cache=value)


def test_release_requires_deferred_setup():
    with pytest.raises(ValueError, match="requires defer_setup"):
        _make_mooncake_store_config(release_model_file_cache=True)
    assert _make_mooncake_store_config(defer_setup=True, release_model_file_cache=True).release_model_file_cache
