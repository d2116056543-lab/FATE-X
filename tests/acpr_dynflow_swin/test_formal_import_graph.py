
import ast
from pathlib import Path


FORBIDDEN = (
    "fate_x.acpr_dynflow",
    "fate_x.acpr_flow",
    "fate_x.acpr_flow_v2",
    "fate_x.models.flowtrace_pmt_model",
    "fate_x.models.token_pmt_adapter",
    "fate_x.models.sinkhorn_transport",
)


def test_formal_namespace_exists_and_avoids_legacy_imports():
    package = Path("fate_x/acpr_dynflow_swin")
    assert package.is_dir(), "formal namespace fate_x/acpr_dynflow_swin is missing"
    py_files = list(package.rglob("*.py"))
    assert py_files, "formal namespace has no python files"
    offenders = []
    for file in py_files:
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(alias.name == bad or alias.name.startswith(bad + ".") for bad in FORBIDDEN):
                        offenders.append((str(file), alias.name))
            elif isinstance(node, ast.ImportFrom) and node.module:
                if any(node.module == bad or node.module.startswith(bad + ".") for bad in FORBIDDEN):
                    offenders.append((str(file), node.module))
    assert not offenders
