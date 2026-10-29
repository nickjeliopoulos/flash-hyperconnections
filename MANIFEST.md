# Repository manifest

This inventory lists all paths tracked by this repository's Git index, including
this manifest. Paths are relative to the repository root. Untracked and ignored
files are excluded. The `third_party/cutlass` submodule is listed once; its
contents are tracked by its own repository.

After adding, moving, or removing tracked files, stage those path changes and run
the following command from the repository root, then stage `MANIFEST.md`:

```bash
python -S - <<'PY'
import subprocess
from pathlib import Path

manifest = Path("MANIFEST.md")
header = manifest.read_text().split("## Tracked files\n", 1)[0]
paths = subprocess.check_output(
    ["git", "ls-files", "--cached", "--deduplicate", "-z"]
).decode().split("\0")
inventory = "\n".join(sorted(path for path in paths if path))
manifest.write_text(header + "## Tracked files\n\n```text\n" + inventory + "\n```\n")
PY
```

## Tracked files

```text
.agents/forum/.gitkeep
.agents/skills/kernel-forum/SKILL.md
.agents/skills/kernel-forum/references/experiment-record.md
.agents/skills/kernel-forum/scripts/forum
.agents/skills/kernel-forum/scripts/test_forum.py
.agents/skills/kernel-ptx-match-forum/SKILL.md
.agents/skills/torch-integrate-review/SKILL.md
.agents/skills/torch-integrate-review/references/integration-checklist.md
.gitignore
.gitmodules
LICENSE
MANIFEST.in
MANIFEST.md
README.md
benchmarks/README.md
benchmarks/__init__.py
benchmarks/common.py
benchmarks/layers/.gitkeep
benchmarks/modules/.gitkeep
benchmarks/ops/matrix_exp.py
benchmarks/ops/stream_mix.py
benchmarks/pytest_gate.py
benchmarks/run.py
benchmarks/visualization.py
csrc/MatrixExpSmall.cuh
csrc/matrix_exp.cu
csrc/register.cpp
csrc/stream_mix.cu
docs/AGENT_GOVERNANCE.md
docs/VALIDATION.md
flash_hypercons/__init__.py
flash_hypercons/ops/__init__.py
flash_hypercons/ops/matrix_exp.py
flash_hypercons/ops/stream_mix.py
flash_hypercons/testing.py
provenance.json
pyproject.toml
setup.py
tests/__init__.py
tests/conftest.py
tests/reference.py
tests/test_benchmark_pipeline.py
tests/test_benchmarks.py
tests/test_integration.py
tests/test_matrix_exp.py
tests/test_routing.py
tests/test_stream_mix.py
tests/test_testing.py
third_party/cutlass
```
