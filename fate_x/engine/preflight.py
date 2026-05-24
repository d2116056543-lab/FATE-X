from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs/fate_x/preflight.json")
    args = ap.parse_args()
    report = {}
    try:
        import torch
        report["torch"] = torch.__version__
        report["cuda"] = torch.cuda.is_available()
        report["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:
        report["torch_error"] = repr(exc)
    for p in ["datasets", "datasets_part", "models", "checkpoints"]:
        report[p] = os.path.exists(p)
    report["datasets_part_h5_count"] = len(glob.glob("datasets_part/BDDX/processed_video_info/*.h5"))
    report["train_yaml"] = os.path.exists("datasets_part/BDDX/training_32frames.yaml")
    report["test_yaml"] = os.path.exists("datasets_part/BDDX/testing_32frames.yaml")
    report["bert_base"] = os.path.exists("models/captioning/bert-base-uncased")
    report["video_swin_k600"] = os.path.exists("models/video_swin_transformer/swin_base_patch244_window877_kinetics600_22k.pth")
    report["adapt_basemodel"] = os.path.exists("checkpoints/basemodel/checkpoints/model.bin")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
