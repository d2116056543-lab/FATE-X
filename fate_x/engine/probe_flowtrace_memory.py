from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from fate_x.engine.flowtrace_adapt_bridge import build_adapt_train_command


DEFAULT_CANDIDATES = [(4, 16), (3, 22), (2, 32)]


def _cuda_memory() -> dict[str, float | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"allocated_gib": None, "reserved_gib": None}
        return {
            "allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        }
    except Exception:
        return {"allocated_gib": None, "reserved_gib": None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flowtrace_pmt_v1_bddx_32f_224.yaml")
    parser.add_argument("--output", default=".background_runs/flowtrace_pmt_v1_preflight/memory_probe.json")
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    selected = None
    for batch_size, accum in DEFAULT_CANDIDATES:
        run_dir = output.parent / f"memory_probe_b{batch_size}_a{accum}"
        command = build_adapt_train_command(
            args.config,
            run_dir,
            epochs=1,
            micro_batch=batch_size,
            grad_accum=accum,
            max_steps=args.iterations,
            beam_size=1,
        )
        command.write_manifest(run_dir)
        started = time.time()
        proc = subprocess.run(command.command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        record = {
            "batch_size": batch_size,
            "gradient_accumulation_steps": accum,
            "effective_batch": batch_size * accum,
            "returncode": proc.returncode,
            "elapsed_seconds": time.time() - started,
            "log_path": str(run_dir / "memory_probe_subprocess.log"),
            "memory": _cuda_memory(),
        }
        (run_dir / "memory_probe_subprocess.log").write_text(proc.stdout, encoding="utf-8", errors="replace")
        results.append(record)
        if proc.returncode == 0 and selected is None:
            selected = record
            break

    report = {"candidates": results, "selected": selected, "formal_losses": True, "direct_image": True}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(str(output))
    if selected is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
