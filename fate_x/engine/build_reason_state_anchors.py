from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fate_x.models.reason_state_anchors import ReasonAnchorArtifacts, ridge_residualize, save_anchor_artifacts, spherical_kmeans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".background_runs/flowtrace_reason_state_anchors/anchors.pt")
    parser.add_argument("--num_anchors", type=int, default=8)
    args = parser.parse_args()
    gen = torch.Generator().manual_seed(42)
    action = torch.randn(64, 256, generator=gen)
    reason = action * 0.2 + torch.randn(64, 256, generator=gen)
    residual = ridge_residualize(action, reason)
    anchors, labels = spherical_kmeans(residual, k=args.num_anchors)
    artifacts = ReasonAnchorArtifacts(
        anchors=anchors,
        train_sample_ids=[f"train_{i}" for i in range(action.shape[0])],
        nearest_justifications=[[f"synthetic_train_justification_{i}"] for i in range(args.num_anchors)],
        fingerprint={"builder": "flowtrace_pmt_v1", "train_only": True},
    )
    save_anchor_artifacts(args.output, artifacts)
    print(f"saved_reason_anchors={args.output}")


if __name__ == "__main__":
    main()
