import json
from pathlib import Path

import torch
from torch import nn

from fate_x.acpr_flow.model import ACPRFlowModel, ACPRFlowModelConfig
from fate_x.acpr_flow.types import ACPRFlowBatch
from fate_x.engine import train_acpr_flowcal_pp


class RecordingCaptioningModel(nn.Module):
    def __init__(self, vocab_size: int = 17) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(768, vocab_size) * 0.01)
        self.calls = []

    def forward(self, **kwargs):
        bundle = kwargs.get("acpr_flow_bundle")
        seca = kwargs.get("acpr_temporal_seca")
        text_len = int(kwargs.get("acpr_text_len") or kwargs["masked_pos"].shape[-1])
        # This is intentionally a small differentiable stand-in for
        # BertForImageCaptioning.encode_forward: it must receive the formal ACPR
        # SECA hook arguments and return a masked loss plus token logits.
        if bundle is None or seca is None:
            enhanced = torch.zeros(kwargs["input_ids"].shape[0], text_len, 768, device=kwargs["input_ids"].device)
        else:
            hidden = bundle.global_reason_state.unsqueeze(1).expand(-1, text_len, -1)
            enhanced, info = seca(
                hidden,
                bundle.reason_memory,
                token_type_ids=kwargs.get("token_type_ids"),
                text_len=text_len,
            )
            bundle.token_reason_attention = info["token_reason_attention"]
            bundle.token_delta = info["token_delta"]
        logits_all = enhanced @ self.weight
        masked_pos = kwargs["masked_pos"].bool()
        logits = logits_all[masked_pos]
        target = kwargs["masked_ids"][masked_pos].clamp_min(0) % logits.shape[-1]
        loss = torch.nn.functional.cross_entropy(logits.float(), target)
        self.calls.append(kwargs)
        return loss, logits


def _tiny_batch(batch_size: int = 2) -> ACPRFlowBatch:
    frames = torch.zeros(batch_size, 32, 3, 64, 64)
    input_ids = torch.ones(batch_size, 8, dtype=torch.long)
    token_type_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]] * batch_size)
    masked_pos = torch.ones(batch_size, 8, dtype=torch.long)
    masked_ids = torch.arange(batch_size * 8, dtype=torch.long).reshape(batch_size, 8)
    return ACPRFlowBatch(
        input_ids=input_ids,
        attention_mask=torch.ones(batch_size, 8 + 16, 8 + 16),
        token_type_ids=token_type_ids,
        frames=frames,
        masked_pos=masked_pos,
        masked_ids=masked_ids,
        car_info=torch.zeros(batch_size, 2, 32),
        sample_ids=[f"sample_{i}" for i in range(batch_size)],
        raw_actions=["slow down"] * batch_size,
        raw_justifications=["front traffic is close"] * batch_size,
    )


def test_formal_forward_routes_text_loss_through_bert_captioning_seca_hook():
    captioning = RecordingCaptioningModel()
    model = ACPRFlowModel(
        ACPRFlowModelConfig(formal_backbone=False),
        captioning_model=captioning,
    )
    batch = _tiny_batch()
    target = torch.nn.functional.normalize(torch.ones(2, 768), dim=-1)

    out = model(batch=batch, reason_semantic_target=target)

    assert len(captioning.calls) >= 2
    enhanced_call = captioning.calls[-1]
    assert enhanced_call["acpr_flow_bundle"] is out.bundle
    assert enhanced_call["acpr_temporal_seca"] is model.temporal_seca
    assert enhanced_call["acpr_text_len"] == batch.masked_pos.shape[-1]
    assert out.bundle.token_reason_attention is not None
    assert out.enhanced_masked_logits.ndim == 2
    assert torch.isfinite(out.action_text_loss)


def test_split_caption_losses_handles_compact_masked_ids_from_bddx_batch():
    model = ACPRFlowModel(ACPRFlowModelConfig(formal_backbone=False))
    logits = torch.randn(3, 17)
    # Real BDD-X batches expose a position mask over the text sequence, while
    # masked_ids is a compact max-masked-token list padded with -1.
    masked_pos = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)
    masked_ids = torch.tensor([[4, 7, -1]], dtype=torch.long)
    token_type_ids = torch.tensor([[0, 1, 1, 1, 1]], dtype=torch.long)

    action_loss, explanation_loss = model._split_caption_losses(
        logits,
        masked_ids,
        masked_pos,
        token_type_ids,
    )

    assert torch.isfinite(action_loss)
    assert torch.isfinite(explanation_loss)


def test_hardpair_loss_is_integrated_into_model_and_optimizer_group():
    captioning = RecordingCaptioningModel()
    model = ACPRFlowModel(
        ACPRFlowModelConfig(formal_backbone=False),
        captioning_model=captioning,
    )
    # Seed an eligible queue entry so the formal path exercises the hard-pair
    # projection and logs the budgeted contribution.
    batch = _tiny_batch(batch_size=1)
    target = torch.nn.functional.normalize(torch.ones(1, 768), dim=-1)
    model.hardpair.enqueue(-target, target)

    out = model(batch=batch, reason_semantic_target=target)
    out.total_loss.backward()

    assert "hardpair_raw_loss" in out.loss_components
    assert "hardpair_budgeted_loss" in out.loss_components
    assert "hardpair_active_pair_rate" in out.loss_components
    assert model.hardpair.proj.weight.grad is not None
    assert torch.isfinite(model.hardpair.proj.weight.grad).all()
    _, manifest = train_acpr_flowcal_pp.build_acpr_optimizer_groups(model)
    assert manifest["hardpair.proj.weight"] == "hardpair_projection"


def test_sequence_calalign_stage_writes_train_calib_only_artifact(tmp_path):
    sample_ids = [f"train_calib_{i}" for i in range(4)]
    base = torch.zeros(4, 3)
    enhanced = torch.randn(4, 3)
    targets = torch.zeros(4, dtype=torch.long)

    artifact = train_acpr_flowcal_pp.run_sequence_calalign_stage(
        output_dir=tmp_path,
        sample_ids=sample_ids,
        base_logits=base,
        enhanced_logits=enhanced,
        targets=targets,
    )

    data = json.loads(Path(artifact).read_text(encoding="utf-8"))
    assert data["fit_split"] == "train_calib"
    assert data["fit_uses_test"] is False
    assert data["zero_alpha_candidate"] is True

def test_model_total_loss_applies_configured_control_loss_weight():
    captioning = RecordingCaptioningModel()
    cfg = ACPRFlowModelConfig(
        formal_backbone=False,
        use_prefix_future=False,
        loss_weights={
            "action_text": 1.0,
            "explanation_text": 1.0,
            "control": 0.0,
            "predicate_pu": 0.0,
            "flow_pu": 0.0,
            "reason_semantic": 0.0,
            "future_control": 0.0,
            "memory_diversity": 0.0,
        },
    )
    model = ACPRFlowModel(cfg, captioning_model=captioning)
    batch = _tiny_batch(batch_size=1)
    batch.car_info = torch.full_like(batch.car_info, 100.0)
    target = torch.nn.functional.normalize(torch.ones(1, 768), dim=-1)

    out = model(batch=batch, reason_semantic_target=target)

    assert out.loss_components["control"].detach().item() > 1000.0
    assert out.loss_components["control_weighted"].detach().item() == 0.0
    assert out.total_loss.detach().item() < out.loss_components["control"].detach().item() * 0.01
