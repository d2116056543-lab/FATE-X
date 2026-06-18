from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from fate_x.engine.flowtrace_adapt_bridge import _load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flowtrace_pmt_v1_bddx_32f_224.yaml")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--output_dir", default=".background_runs/flowtrace_pmt_v1_eval")
    parser.add_argument("--beam_size", type=int, default=3)
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config))
    paths = cfg["paths"]
    checkpoint_dir = args.checkpoint_dir or str(Path(paths["adapt_checkpoint"]).parent)
    test_yaml = str(paths["test_yaml"]).replace("\\", "/")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        "src/tasks/run_adapt.py",
        "--config",
        "src/configs/VidSwinBert/BDDX_multi_default.json",
        "--do_eval",
        "--data_dir",
        ".",
        "--val_yaml",
        test_yaml,
        "--eval_model_dir",
        str(checkpoint_dir).replace("\\", "/") + "/",
        "--model_name_or_path",
        str(paths["bert_dir"]).replace("\\", "/"),
        "--num_beams",
        str(args.beam_size),
        "--use_sep_cap",
        "true",
        "--multitask",
        "true",
        "--signal_types",
        "course",
        "speed",
        "--flowtrace_enabled",
        "true",
        "--flowtrace_state_dim",
        str(cfg.get("model", {}).get("state_dim", 256)),
        "--flowtrace_pmt_rank",
        str(cfg.get("model", {}).get("token_pmt", {}).get("rank", 32)),
        "--fate_x_enabled",
        "true",
        "--video_token_reducer",
        "none",
        "--temporal_evidence_memory",
        "none",
        "--fate_x_reduce_control",
        "false",
    ]
    (output_dir / "eval_command.json").write_text(json.dumps({"command": cmd}, indent=2), encoding="utf-8")
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output_dir / "eval.log").write_text(proc.stdout, encoding="utf-8", errors="replace")
    print(proc.stdout)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
