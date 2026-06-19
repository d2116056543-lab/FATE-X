from __future__ import annotations

import argparse

import torch

from fate_x.acpr_flow.sequence_calalign import SequenceCalAlign
from fate_x.utils.acpr_flow_artifacts import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    ids = [f"train_calib_{i}" for i in range(8)]
    fitter = SequenceCalAlign(ids)
    scales = fitter.fit(ids, torch.zeros(8, 3), torch.randn(8, 3), torch.zeros(8, dtype=torch.long))
    write_json(f"{args.output_dir}/sequence_calalign_audit.json", scales.__dict__ | {"fit_uses_test": fitter.fit_uses_test})


if __name__ == "__main__":
    main()
