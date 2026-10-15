"""Install the scoped engine patch and rebuild only when native sources differ."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

repo = Path("/vllm-workspace/vllm-ascend")
before = json.loads(Path("/opt/qc-native-base.json").read_text())
changed = [
    name
    for name, digest in before.items()
    if not (repo / name).is_file() or hashlib.sha256((repo / name).read_bytes()).hexdigest() != digest
]
print("Changed native base sources:", changed, flush=True)
env = dict(
    os.environ,
    COMPILE_CUSTOM_KERNELS="1" if changed else "0",
    SOC_VERSION="ascend910b1",
    MAX_JOBS="12",
    CMAKE_BUILD_PARALLEL_LEVEL="12",
    SETUPTOOLS_SCM_PRETEND_VERSION="0.31.0.dev0",
)
subprocess.run(
    ["python3", "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", str(repo)], env=env, check=True
)
subprocess.run(
    ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(repo / "vllm_ascend/patch/platform/v41_prefill_barrier.patch")],
    cwd="/vllm-workspace/vllm",
    check=True,
)
subprocess.run(
    [
        "patch",
        "--batch",
        "--fuzz=0",
        "-p1",
        "-i",
        str(repo / "vllm_ascend/patch/platform/v41_repetition_defaults.patch"),
    ],
    cwd="/vllm-workspace/vllm",
    check=True,
)
subprocess.run(["python3", str(repo / "tools/qc_dsv41/build_joint.py")], env=env, check=True)
