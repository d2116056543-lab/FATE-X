from __future__ import annotations

import argparse

from fate_x.explain.acpr_flow_atlas import build_acpr_flow_atlas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    print(build_acpr_flow_atlas([{"sample_id": "smoke", "source": "acpr_flow"}], args.output_dir))


if __name__ == "__main__":
    main()
