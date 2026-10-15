"""Build the private mHC kernel and its binding against the active nightly ABI."""

import os
import shutil
import subprocess
from pathlib import Path

import torch_npu
from torch.utils.cpp_extension import load

repo = Path("/vllm-workspace/vllm-ascend")
output = Path("/opt/v41hcpostprenorm")
output.mkdir(exist_ok=True)
source = output / "build/csrc"
source.parent.mkdir(exist_ok=True)
if source.exists():
    shutil.rmtree(source)


def ignore_build_outputs(directory, names):
    # Only top-level build outputs are disposable. In particular, nlohmann's
    # include/nlohmann/detail/output directory contains required source headers.
    if Path(directory) == repo / "csrc":
        return {"build", "output", "build_out"} & set(names)
    return {"__pycache__"} & set(names)


shutil.copytree(repo / "csrc", source, ignore=ignore_build_outputs)
subprocess.run(
    [
        "bash",
        "build.sh",
        "--pkg",
        "--soc=ascend910b",
        "--ops=v41_hc_post_pre_norm",
        "--vendor_name=v41hcpostprenorm",
        "-j8",
    ],
    cwd=source,
    check=True,
)
packages = [p for p in source.glob("build/**/*.run") if "_CPack_Packages" not in p.parts and p.stat().st_size > 100000]
if len(packages) != 1:
    raise RuntimeError(f"Expected one private vendor package, got {packages}")
installed = output / "installed"
installed.mkdir(exist_ok=True)
subprocess.run(["bash", str(packages[0]), "--quiet", "--install-path=" + str(installed)], check=True)
npu = Path(torch_npu.__file__).parent
ascend = Path(os.environ["ASCEND_HOME_PATH"])
adapter = repo / "csrc/aclnn_torch_adapter"
binding = output / "binding"
binding.mkdir(exist_ok=True)
library = load(
    name="v41_hc_post_pre_norm_native",
    sources=[str(adapter / n) for n in ["v41_hc_post_pre_norm_binding.cpp", "NPUBridge.cpp", "NPUStorageImpl.cpp"]],
    extra_include_paths=[str(repo / "csrc"), str(adapter), str(npu / "include"), str(ascend / "include")],
    extra_cflags=["-O2", "-fvisibility=hidden"],
    extra_ldflags=["-L" + str(npu / "lib"), "-ltorch_npu", "-L" + str(ascend / "lib64"), "-lascendcl"],
    build_directory=str(binding),
    is_python_module=False,
    verbose=True,
)
shutil.copyfile(library, output / "native.so")
vendors = list(installed.glob("vendors/*"))
if len(vendors) != 1:
    raise RuntimeError(f"Expected one installed vendor, got {vendors}")
(output / "vendor").symlink_to(vendors[0])
