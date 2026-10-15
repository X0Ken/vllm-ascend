"""Build 0.31 Rust components without reusing newer nightly workspace artifacts."""

import os
import subprocess
import sys
from pathlib import Path

repo = Path("/vllm-workspace/vllm")
# Git archives retain release timestamps, which predate cached nightly objects.
# Refresh local source timestamps so Cargo checks every workspace crate again.
for path in (repo / "rust").rglob("*"):
    if (
        "target" not in path.parts
        and path.is_file()
        and (path.suffix in {".rs", ".proto"} or path.name == "Cargo.toml")
    ):
        path.touch()
env = dict(os.environ, VLLM_RS_BUILD_VERSION="0.31.0", SETUPTOOLS_SCM_PRETEND_VERSION="0.31.0", CARGO_BUILD_JOBS="12")
env["PATH"] = "/root/.cargo/bin:" + env.get("PATH", "")
subprocess.run([sys.executable, str(repo / "tools/build_rust.py"), "--release"], cwd=repo, env=env, check=True)
