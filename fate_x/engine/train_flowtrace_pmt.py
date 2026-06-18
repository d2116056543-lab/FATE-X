from __future__ import annotations

import argparse
import json
from pathlib import Path

from fate_x.engine.flowtrace_adapt_bridge import build_adapt_train_command, run_attached


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flowtrace_pmt_v1_bddx_32f_224.yaml")
    parser.add_argument("--output_dir", default=".background_runs/flowtrace_pmt_v1_formal")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--micro_batch", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=None, help="Only for real-data smoke; limits samples via ADAPT limited_samples.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Only for real-data smoke; limits ADAPT evaluation samples.")
    parser.add_argument("--beam_size", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_adapt_train_command(
        args.config,
        output_dir,
        epochs=args.epochs,
        micro_batch=args.micro_batch,
        grad_accum=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        max_eval_samples=args.max_eval_samples,
        beam_size=args.beam_size,
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "engine": "flowtrace_pmt_v1_real_adapt_direct_image",
                "formal_trainer": True,
                "feature_cache_enabled": False,
                "token_cache_enabled": False,
                "command": command.command,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    raise SystemExit(run_attached(command, output_dir))


if __name__ == "__main__":
    main()
