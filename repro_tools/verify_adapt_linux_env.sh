#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/mnt/e/sbw/ADAPT_repro/ADAPT/repro_logs/linux_setup"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/verify_adapt_linux_env.log") 2>&1

echo "==== ADAPT Linux verification started: $(date -Is) ===="
source /opt/conda/etc/profile.d/conda.sh
conda activate adapt
cd /mnt/e/sbw/ADAPT_repro/ADAPT

python - <<'PY'
import os
from pathlib import Path
print("python ok")
print("repo", Path.cwd())
print("data path exists", Path("/mnt/e/sbw/ADAPT_repro/ADAPT/datasets/BDDX").exists())
print("checkpoint exists", Path("/mnt/e/sbw/ADAPT_repro/ADAPT/checkpoints/basemodel/checkpoints/model.bin").exists())
print("swin exists", Path("/mnt/e/sbw/ADAPT_repro/ADAPT/models/video_swin_transformer/swin_base_patch244_window877_kinetics600_22k.pth").exists())
PY

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY

python - <<'PY'
import transformers
import deepspeed
from apex import amp
print("transformers", transformers.__version__)
print("deepspeed import ok")
print("apex amp import ok")
PY

python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from src.configs.config import shared_configs
print("ADAPT config import ok")
PY

echo "==== ADAPT Linux verification finished: $(date -Is) ===="
