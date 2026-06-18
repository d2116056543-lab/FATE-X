from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=".background_runs/flowtrace_pmt_v1_eval")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {"caption_metrics_action": {"unavailable_reason": "requires formal ADAPT decoder checkpoint"},
              "caption_metrics_explanation": {"unavailable_reason": "requires formal ADAPT decoder checkpoint"}}
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
