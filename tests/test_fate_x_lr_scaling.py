from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from fate_x.engine.lr_scaling import apply_lr_scaling_to_args, compute_lr_scaling, effective_batch_size


def test_fate_x_effective_batch_linear_lr_scaling():
    assert effective_batch_size(2, 32, 1) == 64
    result = compute_lr_scaling(
        per_gpu_batch_size=2,
        gradient_accumulation_steps=32,
        reference_effective_batch=64,
        base_learning_rate_at_reference_batch=2e-4,
        backbone_coef_lr=0.05,
        auto_scale_lr=True,
        current_lr=1e-4,
    )
    assert result.effective_batch_size == 64
    assert result.lr_actual == 2e-4
    assert result.backbone_lr == 1e-5


def test_apply_lr_scaling_to_args_sets_manifest_fields():
    args = Namespace(
        per_gpu_train_batch_size=1,
        gradient_accumulation_steps=32,
        reference_effective_batch=64,
        base_learning_rate_at_reference_batch=2e-4,
        backbone_coef_lr=0.05,
        num_gpus_for_lr=1,
        auto_scale_lr=True,
        learning_rate=2e-4,
        mixed_precision_method="apex",
    )
    apply_lr_scaling_to_args(args)
    assert args.effective_batch_size == 32
    assert args.learning_rate == 1e-4
    assert args.backbone_lr == 5e-6
    assert args.loss_divided_by_accumulation is True
    assert args.accumulation_handled_by == "manual"


def test_run_adapt_divides_manual_gradient_accumulation_loss():
    source = Path("src/tasks/run_adapt.py").read_text(encoding="utf-8")
    assert "loss_for_backward = loss if args.mixed_precision_method == \"deepspeed\" else loss / float(args.gradient_accumulation_steps)" in source
    assert "scaler.scale(loss_for_backward).backward()" in source
    assert "amp.scale_loss(loss_for_backward" in source
