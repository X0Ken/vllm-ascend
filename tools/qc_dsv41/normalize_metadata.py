"""Remove nightly metadata that shadows the installed editable distribution."""

import importlib.metadata as metadata
import runpy
import shutil
import sysconfig
from pathlib import Path

repo = Path("/vllm-workspace/vllm-ascend")
site_packages = Path(sysconfig.get_path("purelib"))
expected = "0.31.0.dev0"
for root in (repo, site_packages):
    for distribution in metadata.distributions(path=[str(root)]):
        if distribution.metadata.get("Name", "").replace("_", "-").lower() != "vllm-ascend":
            continue
        path = Path(distribution._path).resolve()
        if distribution.version != expected:
            assert path.parent == root.resolve()
            assert path.name.startswith("vllm_ascend") and path.suffix in (".egg-info", ".dist-info")
            shutil.rmtree(path)
assert metadata.version("vllm-ascend") == expected
assert runpy.run_path(str(repo / "vllm_ascend/_version.py"))["__version__"] == expected
