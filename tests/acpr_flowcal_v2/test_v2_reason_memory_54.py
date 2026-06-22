from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from _v2_contract_smoke import make_reason_memory


def test_reason_memory_contains_54_typed_tokens():
    memory = make_reason_memory(dim=8)
    assert memory.values.shape[1] == 54
    assert len(memory.names) == 54
