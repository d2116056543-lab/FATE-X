from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".background_runs/flowtrace_pmt_v1_preflight/memory_probe.json")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"selected_batch_size": 2, "gradient_accumulation_steps": 32,
                                             "note": "synthetic probe placeholder until formal data paths are linked"}, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
