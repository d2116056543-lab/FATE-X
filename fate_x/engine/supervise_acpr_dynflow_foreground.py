from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--max_steps", type=int, default=-1)
    args = p.parse_args()
    cmd = [sys.executable, "-m", "fate_x.engine.train_acpr_dynflow", "--config", args.config, "--output_dir", args.output_dir, "--device", args.device]
    if args.synthetic:
        cmd.append("--synthetic")
    if args.max_steps > 0:
        cmd += ["--max_steps", str(args.max_steps)]
    print("ACPR_DYNFLOW_FOREGROUND_SUPERVISOR " + " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

