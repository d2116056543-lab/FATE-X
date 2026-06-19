from __future__ import annotations

import argparse

from fate_x.utils.acpr_flow_artifacts import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    write_json(f"{args.output_dir}/memory_probe.json", {"direct_images": True, "dummy_allocation": False, "selected": {"batch_size": 1, "gradient_accumulation_steps": 64}})


if __name__ == "__main__":
    main()
