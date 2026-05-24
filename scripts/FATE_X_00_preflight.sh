#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python - <<'PY'
import json, os, glob
report = {}
try:
    import torch
    report["torch"] = torch.__version__
    report["cuda"] = torch.cuda.is_available()
    report["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except Exception as e:
    report["torch_error"] = repr(e)
for p in ["datasets", "datasets_part", "models", "checkpoints"]:
    report[p] = os.path.exists(p)
report["datasets_part_h5_count"] = len(glob.glob("datasets_part/BDDX/processed_video_info/*.h5"))
report["train_yaml"] = os.path.exists("datasets_part/BDDX/training_32frames.yaml")
report["test_yaml"] = os.path.exists("datasets_part/BDDX/testing_32frames.yaml")
report["bert_base"] = os.path.exists("models/captioning/bert-base-uncased")
report["video_swin_k600"] = os.path.exists("models/video_swin_transformer/swin_base_patch244_window877_kinetics600_22k.pth")
report["adapt_basemodel"] = os.path.exists("checkpoints/basemodel/checkpoints/model.bin")
os.makedirs("outputs/fate_x", exist_ok=True)
open("outputs/fate_x/preflight.json","w").write(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
PY
