from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=False)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "comparison_only": True,
        "checkpoint": args.checkpoint,
        "checkpoint_exists": bool(args.checkpoint and Path(args.checkpoint).exists()),
        "training_dependency_forbidden": True,
        "metrics_available": False,
        "blocker": None if args.checkpoint and Path(args.checkpoint).exists() else "ADAPT reference checkpoint unresolved",
    }
    (out / "adapt_reference_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "adapt_reference_predictions_manifest.json").write_text(json.dumps({"predictions_used_for_training": False}, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

