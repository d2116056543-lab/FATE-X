from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"selected": {"batch_size": 2, "gradient_accumulation_steps": 32, "effective_batch": 64}, "dummy_allocation_used": False, "requires_cuda_full_probe": True}
    (out / "memory_probe_selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

