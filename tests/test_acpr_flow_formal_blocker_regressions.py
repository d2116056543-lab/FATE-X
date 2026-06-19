import inspect

import torch

from fate_x.acpr_flow.model import ACPRFlowModel, ACPRFlowModelConfig, TinyDirectImageVideoBackbone
from fate_x.acpr_flow.types import ACPRFlowBatch
from fate_x.engine import train_acpr_flowcal_pp


def test_formal_model_config_uses_non_tiny_backbone_by_default():
    cfg = ACPRFlowModelConfig(formal_backbone=True, load_pretrained_backbone=False)
    model = ACPRFlowModel(cfg)
    assert not isinstance(model.backbone, TinyDirectImageVideoBackbone)
    assert getattr(model.backbone, "formal_backbone_name", "") == "adapt_video_swin_multiscale"


def test_formal_trainer_source_uses_real_bddx_batch_path_not_random_smoke():
    source = inspect.getsource(train_acpr_flowcal_pp)
    assert "torch.randn" not in source
    assert "train_smoke" not in source
    assert "build_bddx_acpr_dataloader" in source
    assert "adapt_checkpoint" in source
    assert "video_swin_checkpoint" in source


def test_formal_forward_wires_pu_reason_and_future_losses():
    cfg = ACPRFlowModelConfig(formal_backbone=False, use_flow=True, use_prefix_future=True)
    model = ACPRFlowModel(cfg)
    frames = torch.zeros(2, 32, 3, 64, 64)
    batch = ACPRFlowBatch(
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        frames=frames,
        masked_ids=torch.ones(2, 16, dtype=torch.long),
        car_info=torch.zeros(2, 2, 32),
        sample_ids=["a", "b"],
        raw_actions=["the car slows down", "the car stops"],
        raw_justifications=["cars ahead are stopped", "a pedestrian is in the crosswalk"],
    )
    reason_target = torch.nn.functional.normalize(torch.ones(2, 768), dim=-1)
    output = model(batch=batch, reason_semantic_target=reason_target)
    for key in ("predicate_pu", "flow_pu", "reason_semantic", "future_control"):
        assert key in output.loss_components
        assert torch.isfinite(output.loss_components[key])
        assert output.loss_components[key].detach().abs().item() > 0


def test_formal_trainer_builds_online_reason_semantic_target_from_raw_text():
    class TinyTokenizer:
        cls_token = "[CLS]"
        sep_token = "[SEP]"
        pad_token = "[PAD]"
        mask_token = "[MASK]"

        vocab = {
            "[PAD]": 0,
            "[CLS]": 1,
            "[SEP]": 2,
            "[MASK]": 3,
            "slow": 4,
            "down": 5,
            "cars": 6,
            "ahead": 7,
            "stopped": 8,
            "pedestrian": 9,
            "crosswalk": 10,
        }

        def tokenize(self, text):
            return str(text).lower().split()

        def convert_tokens_to_ids(self, tokens):
            return [self.vocab.get(token, 0) for token in tokens]

    embedding = torch.arange(11 * 8, dtype=torch.float32).reshape(11, 8) / 100.0
    target = train_acpr_flowcal_pp.build_reason_semantic_target_for_batch(
        raw_actions=["slow down"],
        raw_justifications=["cars ahead stopped"],
        tokenizer=TinyTokenizer(),
        word_embedding_weight=embedding,
        device="cpu",
    )
    assert target.shape == (1, 8)
    assert target.requires_grad is False
    assert torch.isfinite(target).all()
    assert target.detach().abs().sum().item() > 0
    source = inspect.getsource(train_acpr_flowcal_pp.train_formal)
    assert "build_reason_semantic_target_for_batch" in source
    assert "reason_semantic_target=reason_semantic_target" in source


def test_formal_suite_plan_contains_required_stages():
    stages = train_acpr_flowcal_pp.build_formal_experiment_suite(
        "configs/acpr_flowcal_pp_v1_bddx_32f_224.yaml"
    )
    names = [stage["name"] for stage in stages]
    assert names == [
        "run0_adapt_baseline_eval",
        "common_stage_a_acpr_x",
        "fork_b1_acpr_x_equal_budget",
        "fork_b2_acpr_flowcal_pp",
        "sequence_calalign_b1",
        "sequence_calalign_b2",
        "no_retrain_interventions",
        "canvas_generation",
        "dataset_atlas",
    ]
    assert all(stage["metric_based_early_stop"] is False for stage in stages)
