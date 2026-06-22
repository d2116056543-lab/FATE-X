from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--require_review_pass", default="REVIEW_PASS_ACPR_FLOWCAL_V2.txt")
    parser.add_argument("--extra", nargs="*", default=[])
    args = parser.parse_args()
    if args.require_review_pass and not Path(args.require_review_pass).exists():
        raise SystemExit(f"Missing required review pass: {args.require_review_pass}")
    cmd = [
        sys.executable,
        "-m",
        "fate_x.engine.train_acpr_flowcal_v2",
        "--config",
        args.config,
        "--output_dir",
        args.output_dir,
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--epochs",
        str(args.epochs),
    ] + list(args.extra)
    print("ACPR_FLOWCAL_V2_SUPERVISOR_CMD " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd)
    raise SystemExit(proc.wait())


if __name__ == "__main__":
    main()
