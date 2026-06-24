from __future__ import annotations

import ast
import argparse
import json
from pathlib import Path


FORBIDDEN = (
    "fate_x.acpr_dynflow",
    "fate_x.acpr_flow",
    "fate_x.acpr_flow_v2",
    "fate_x.models.flowtrace_pmt_model",
    "fate_x.models.token_pmt_adapter",
    "fate_x.models.sinkhorn_transport",
)


def audit_import_graph(package: str = "fate_x/acpr_dynflow_swin") -> dict:
    offenders = []
    for file in Path(package).rglob("*.py"):
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom):
                module = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if any(alias.name == bad or alias.name.startswith(bad + ".") for bad in FORBIDDEN):
                        offenders.append({"file": str(file), "module": alias.name})
            if module and any(module == bad or module.startswith(bad + ".") for bad in FORBIDDEN):
                offenders.append({"file": str(file), "module": module})
    return {"passed": not offenders, "offenders": offenders}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--package", default="fate_x/acpr_dynflow_swin")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    result = audit_import_graph(args.package)
    if args.config is not None:
        result["config"] = args.config
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "audit_summary.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
