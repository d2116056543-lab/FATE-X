import torch

from fate_x.acpr_flow_v2.temporal_seca import TemporalSECAV2
from fate_x.acpr_flow_v2.types import SemanticReasonMemory


def test_temporal_seca_reads_memory_and_returns_attention():
    hidden = torch.randn(2, 5, 16)
    memory = SemanticReasonMemory(
        values=torch.randn(2, 54, 16),
        mask=torch.ones(2, 54, dtype=torch.bool),
        confidence=torch.ones(2, 54),
        names=tuple(str(i) for i in range(54)),
        type_ids=torch.zeros(54, dtype=torch.long),
        axis_ids=torch.zeros(54, dtype=torch.long),
        evidence_maps=torch.zeros(2, 1, 54, 2, 2),
        lineage=[],
        semantic_state=torch.randn(2, 16),
    )
    out, diag = TemporalSECAV2(hidden_dim=16)(hidden, memory, None, 5, generation_segment="explanation")
    assert out.shape == hidden.shape
    assert diag.attention.shape == (2, 5, 54)


def test_temporal_seca_explanation_segment_does_not_rewrite_description_or_image_tokens():
    hidden = torch.randn(1, 8, 16)
    memory = SemanticReasonMemory(
        values=torch.randn(1, 54, 16),
        mask=torch.ones(1, 54, dtype=torch.bool),
        confidence=torch.ones(1, 54),
        names=tuple(str(i) for i in range(54)),
        type_ids=torch.zeros(54, dtype=torch.long),
        axis_ids=torch.zeros(54, dtype=torch.long),
        evidence_maps=torch.zeros(1, 1, 54, 2, 2),
        lineage=[],
        semantic_state=torch.randn(1, 16),
    )
    token_type_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])
    seca = TemporalSECAV2(hidden_dim=16)
    with torch.no_grad():
        seca.gate_explanation.fill_(1.0)
        seca.gate_action.fill_(0.0)
    out, diag = seca(hidden, memory, token_type_ids=token_type_ids, text_len=6)
    assert torch.allclose(out[:, :3], hidden[:, :3])
    assert not torch.allclose(out[:, 3:6], hidden[:, 3:6])
    assert torch.allclose(out[:, 6:], hidden[:, 6:])
    assert float(diag.image_hidden_max_diff) == 0.0
