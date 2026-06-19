from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


FORBIDDEN_DETACH = ["Start-Process", "Start-Job", "schtasks", "nohup", "DETACHED_PROCESS", "-WindowStyle Hidden"]


def scan_no_detach(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    return [x for x in FORBIDDEN_DETACH if x in text]


def write_foreground_supervisor_smoke(
    output_dir: str | Path,
    command: list[str],
    heartbeat_seconds: int,
    child_pid: int | None = None,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "attached_foreground": True,
        "detached_process": False,
        "metric_based_stop": False,
        "parent_pid": os.getpid(),
        "child_pid": child_pid,
        "command": list(command),
        "heartbeat_seconds": int(heartbeat_seconds),
        "created_at_unix": time.time(),
    }
    artifact = out / "foreground_supervisor_smoke.json"
    artifact.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", nargs="+", required=True)
    parser.add_argument("--heartbeat_seconds", type=int, default=60)
    parser.add_argument("--output_dir", default=".background_runs/acpr_flowcal_pp_v1_preflight")
    args = parser.parse_args()
    proc = subprocess.Popen(args.command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    write_foreground_supervisor_smoke(args.output_dir, args.command, args.heartbeat_seconds, child_pid=proc.pid)
    last = time.time()
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        if time.time() - last >= args.heartbeat_seconds:
            print(f"ACPR_SUPERVISOR_HEARTBEAT pid={proc.pid}", flush=True)
            last = time.time()
    sys.exit(proc.wait())


if __name__ == "__main__":
    main()
