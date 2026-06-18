from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


class FlowTraceForegroundSupervisor:
    def __init__(
        self,
        config: str,
        output_dir: str,
        *,
        require_review_pass: bool = True,
        heartbeat_seconds: int = 60,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.require_review_pass = require_review_pass
        self.heartbeat_seconds = int(heartbeat_seconds)

    def _verify_review_pass(self) -> None:
        pass_file = Path(".background_runs/flowtrace_pmt_v1_preflight/REVIEW_PASS_FLOWTRACE_PMT_V1.txt")
        if self.require_review_pass and not pass_file.exists():
            raise SystemExit("REVIEW_PASS_FLOWTRACE_PMT_V1.txt is required before formal training.")

    def run(self) -> int:
        self._verify_review_pass()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        status_path = self.output_dir / "supervisor_live_status.json"
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "fate_x.engine.train_flowtrace_pmt",
            "--config",
            self.config,
            "--output_dir",
            str(self.output_dir),
        ]
        proc = subprocess.Popen(cmd)
        while proc.poll() is None:
            status = {
                "heartbeat": time.time(),
                "child_pid": proc.pid,
                "attached_child": True,
                "detached": False,
                "command": cmd,
            }
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            print("FLOWTRACE_SUPERVISOR_HEARTBEAT " + json.dumps(status), flush=True)
            time.sleep(self.heartbeat_seconds)
        status = {
            "heartbeat": time.time(),
            "child_pid": proc.pid,
            "returncode": int(proc.returncode),
            "attached_child": True,
            "detached": False,
            "command": cmd,
        }
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        return int(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flowtrace_pmt_v1_bddx_32f_224.yaml")
    parser.add_argument("--output_dir", default=".background_runs/flowtrace_pmt_v1_formal")
    parser.add_argument("--require_review_pass", action="store_true")
    parser.add_argument("--heartbeat_seconds", type=int, default=60)
    args = parser.parse_args()
    raise SystemExit(
        FlowTraceForegroundSupervisor(
            args.config,
            args.output_dir,
            require_review_pass=args.require_review_pass,
            heartbeat_seconds=args.heartbeat_seconds,
        ).run()
    )


if __name__ == "__main__":
    main()
