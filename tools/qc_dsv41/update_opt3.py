"""Apply the three Python optimizations to the qualified P10 runtime."""

import hashlib
import json
import subprocess
from pathlib import Path

repo = Path("/vllm-workspace/vllm-ascend")
before = json.loads(Path("/opt/qc-opt3-native-base.json").read_text())
for name, digest in before.items():
    assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest, f"Native source changed: {name}"
patch = repo / "vllm_ascend/patch/platform/v41_opt3.patch"
command = ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(patch)]
subprocess.run([*command, "--dry-run"], cwd="/vllm-workspace/vllm", check=True)
subprocess.run(command, cwd="/vllm-workspace/vllm", check=True)
# Existing editable installations, Rust renderer and native binaries stay intact.
