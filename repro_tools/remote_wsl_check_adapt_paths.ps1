$ErrorActionPreference = "Stop"
wsl.exe -d ADAPT-Ubuntu -- bash -lc "set -e; cd /mnt/e/sbw/ADAPT_repro/ADAPT; echo '---wsl path check---'; ls -l datasets datasets_part | head -40; test -f datasets/BDDX/training_32frames.yaml; test -f datasets_part/BDDX/training_32frames.yaml; python - <<'PY'
from pathlib import Path
for p in ['datasets/BDDX/training_32frames.yaml','datasets_part/BDDX/training_32frames.yaml','models/captioning/bert-base-uncased/config.json','checkpoints/basemodel/checkpoints/model.bin']:
    path=Path(p)
    print(p, path.exists(), path.stat().st_size if path.exists() else None)
PY"
