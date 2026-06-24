
import torch


def test_formal_dataclasses_and_forward_contract():
    from fate_x.acpr_dynflow_swin.model import ACPRDynFlowSwinModel
    from fate_x.acpr_dynflow_swin.types import DynFlowSwinBatch, ACPRDynFlowSwinOutput

    model = ACPRDynFlowSwinModel({"model": {"dimensions": {"predicate": 32, "traffic": 32, "text": 64, "motion": 64}}})
    batch = DynFlowSwinBatch(
        frames=torch.randn(2, 32, 3, 32, 32),
        input_ids=torch.randint(0, 100, (2, 30)),
        attention_mask=torch.ones(2, 30, dtype=torch.long),
        token_type_ids=torch.zeros(2, 30, dtype=torch.long),
        masked_pos=torch.ones(2, 30, dtype=torch.long),
        masked_ids=torch.randint(0, 100, (2, 30)),
        control_target=torch.randn(2, 32, 2),
        sample_ids=["a", "b"],
        raw_actions=["slow down", "keep moving"],
        raw_justifications=["car ahead", "road clear"],
    )
    out = model(batch)
    assert isinstance(out, ACPRDynFlowSwinOutput)
    assert out.backbone.forward_count == 1
    assert out.predicates.names and len(out.predicates.names) == 32
    assert out.semantic_tokens.assignment.shape[-1] == 5
    assert out.traffic.factor_tokens_native.shape[2] == 13
    assert out.motion.global_prediction_normalized.shape == (2, 32, 2)
    assert out.ledger.final_prediction_normalized.shape == (2, 32, 2)
    assert torch.isfinite(out.total_loss)
    assert set(["action_text", "explanation_text", "final_speed_normalized", "final_course_normalized"]).issubset(out.loss_components)
