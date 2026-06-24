from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acpr_dynflow_swin_v1_bddx_32f_224.yaml")
    parser.add_argument("--require_review_pass", action="store_true")
    args = parser.parse_args()
    pass_file = Path(".background_runs/acpr_dynflow_swin_v1_preflight/REVIEW_PASS_ACPR_DYNFLOW_SWIN_V1.txt")
    if args.require_review_pass and not pass_file.exists():
        raise SystemExit(f"review pass missing: {pass_file}")
    cmd = [sys.executable, "-m", "fate_x.engine.train_acpr_dynflow_swin", "--config", args.config]
    child = subprocess.Popen(cmd)
    while child.poll() is None:
        print(f"dynflow_swin_foreground_heartbeat ts={int(time.time())}", flush=True)
        time.sleep(60)
    raise SystemExit(child.returncode)


if __name__ == "__main__":
    main()
