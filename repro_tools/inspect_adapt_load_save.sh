#!/usr/bin/env bash
set -euo pipefail
cd /mnt/e/sbw/ADAPT_repro/ADAPT
source /opt/conda/etc/profile.d/conda.sh
conda activate adapt
python - <<'PY'
from pathlib import Path
p=Path('src/utils/load_save.py')
lines=p.read_text(encoding='utf-8', errors='replace').splitlines()
for start,end in [(1,260),(260,380)]:
    print(f'--- load_save.py:{start}-{end} ---')
    for i in range(start, min(end, len(lines))+1):
        print(f'{i}: {lines[i-1]}')
PY
