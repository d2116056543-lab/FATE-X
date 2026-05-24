#!/usr/bin/env bash
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate adapt
python - <<'PY'
from pathlib import Path
import torch
root = Path(torch.__file__).resolve().parent
print("torch_root", root)
for p in root.rglob("libcudnn*"):
    print(p)
for p in root.rglob("libcuda*"):
    print(p)
PY
find /opt/conda/envs/adapt -name 'libcudnn*' -o -name 'libcuda*' | head -80
