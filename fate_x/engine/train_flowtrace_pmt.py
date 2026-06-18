from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fate_x.losses.flowtrace_losses import FlowTraceLoss
from fate_x.models.flowtrace_pmt_model import FlowTracePMTModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=".background_runs/flowtrace_pmt_v1_smoke")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = FlowTracePMTModel(fine_dim=32, coarse_dim=64, dense_dim=64, state_dim=32, num_tracks=4, num_states=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = FlowTraceLoss()
    log_path = out / "loss_components.jsonl"
    for epoch in range(args.epochs):
        for step in range(args.max_steps):
            dense = torch.randn(2, 8, 64, device=device)
            fine = torch.randn(2, 32, 4, 6, 6, device=device)
            coarse = torch.randn(2, 64, 4, 3, 3, device=device)
            bundle = model(dense, fine, coarse)
            loss, logs = criterion(bundle)
            loss = loss + bundle.state_memory.pow(2).mean() * 0.01
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            record = {"epoch": epoch, "step": step, "loss": float(loss.detach().cpu()), "batch_size": 2,
                      "state_shape": list(bundle.state_memory.shape)}
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print("FLOWTRACE_BATCH " + json.dumps(record), flush=True)
        ckpt = out / "checkpoint_latest"
        ckpt.mkdir(exist_ok=True)
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt / "model.pt")
        epoch_dir = out / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(exist_ok=True)
        (epoch_dir / "combined_test_score.json").write_text(json.dumps({"unavailable_reason": "synthetic smoke trainer only"}), encoding="utf-8")
    (out / "run_complete.json").write_text(json.dumps({"complete": True, "epochs": args.epochs}), encoding="utf-8")


if __name__ == "__main__":
    main()
