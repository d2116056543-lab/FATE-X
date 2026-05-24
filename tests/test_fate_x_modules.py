import torch

from fate_x.explain.phrase_attribution import find_phrase_hits
from fate_x.models.temporal_evidence_memory import TemporalEvidenceMemory
from fate_x.models.video_token_reducer import VideoTokenReducer


def test_video_token_reducer_shapes_and_provenance():
    x = torch.randn(2, 100, 32)
    r = VideoTokenReducer(32, keep_ratio=0.5, num_summary_tokens=4, min_tokens=16)
    out = r(x)
    assert out["tokens"].shape[0] == 2
    assert out["provenance"].shape[:2] == (2, 100)
    assert torch.allclose(out["provenance"].sum(-1), torch.ones(2, 100), atol=1e-5)


def test_temporal_evidence_memory_shapes():
    x = torch.randn(2, 64, 32)
    mem = TemporalEvidenceMemory(32, num_heads=4)
    out = mem(x)
    assert out["event_tokens"].shape[:2] == (2, 8)


def test_phrase_hits():
    hits = find_phrase_hits("The car stops because the traffic light is red.")
    concepts = {h.concept for h in hits}
    assert "car_vehicle" in concepts
    assert "traffic_light" in concepts
