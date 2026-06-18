from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


class FlowTraceForegroundSupervisor:
    def __init__(self, config: str, require_review_pass: bool = True) -> None:
        self.config = config
        self.require_review_pass = require_review_pass

    def run(self) -> int:
        pass_file = Path(".background_runs/flowtrace_pmt_v1_preflight/REVIEW_PASS_FLOWTRACE_PMT_V1.txt")
        if self.require_review_pass and not pass_file.exists():
            raise SystemExit("REVIEW_PASS_FLOWTRACE_PMT_V1.txt is required before formal training.")
        cmd = [sys.executable, "-u", "-m", "fate_x.engine.train_flowtrace_pmt",
               "--output_dir", ".background_runs/flowtrace_pmt_v1_foreground_smoke",
               "--epochs", "1", "--max_steps", "2"]
        proc = subprocess.Popen(cmd)
        while proc.poll() is None:
            print(json.dumps({"heartbeat": time.time(), "child_pid": proc.pid}), flush=True)
            time.sleep(60)
        return int(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flowtrace_pmt_v1_bddx_32f_224.yaml")
    parser.add_argument("--require_review_pass", action="store_true")
    args = parser.parse_args()
    raise SystemExit(FlowTraceForegroundSupervisor(args.config, args.require_review_pass).run())


if __name__ == "__main__":
    main()
