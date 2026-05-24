#!/usr/bin/env bash
set -euo pipefail
cd /mnt/e/sbw/ADAPT_repro/ADAPT
source /opt/conda/etc/profile.d/conda.sh
conda activate adapt
echo "---SAVE/CKPT LOCATIONS---"
python - <<'PY'
from pathlib import Path
p=Path('src/tasks/run_adapt.py')
lines=p.read_text(encoding='utf-8', errors='replace').splitlines()
for start,end in [(270,360),(1,110),(700,740),(880,910)]:
    print(f'--- run_adapt.py:{start}-{end} ---')
    for i in range(start, min(end, len(lines))+1):
        print(f'{i}: {lines[i-1]}')
print('--- helpers grep targeted ---')
for p in Path('src').rglob('*.py'):
    txt=p.read_text(encoding='utf-8', errors='replace').splitlines()
    for i,l in enumerate(txt,1):
        if 'def save_checkpoint' in l or 'class TrainingRestorer' in l or 'def save(' in l and 'checkpoint' in str(p):
            print(f'{p}:{i}: {l}')
PY
