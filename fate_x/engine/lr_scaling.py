from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LRScalingResult:
    num_gpus: int
    per_gpu_batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    reference_effective_batch: int
    base_lr_at_reference_batch: float
    lr_actual: float
    backbone_coef_lr: float
    backbone_lr: float
    accumulation_handled_by: str


def effective_batch_size(per_gpu_batch_size: int, gradient_accumulation_steps: int, num_gpus: int = 1) -> int:
    return int(max(1, num_gpus) * max(1, per_gpu_batch_size) * max(1, gradient_accumulation_steps))


def scale_lr_linear(base_lr_at_reference_batch: float, effective_batch: int, reference_effective_batch: int) -> float:
    if reference_effective_batch <= 0:
        raise ValueError("reference_effective_batch must be positive")
    return float(base_lr_at_reference_batch) * float(effective_batch) / float(reference_effective_batch)


def compute_lr_scaling(
    *,
    per_gpu_batch_size: int,
    gradient_accumulation_steps: int,
    reference_effective_batch: int = 64,
    base_learning_rate_at_reference_batch: float = 2e-4,
    backbone_coef_lr: float = 0.05,
    num_gpus: int = 1,
    auto_scale_lr: bool = False,
    current_lr: float | None = None,
    mixed_precision_method: str = "",
) -> LRScalingResult:
    eff = effective_batch_size(per_gpu_batch_size, gradient_accumulation_steps, num_gpus)
    lr = (
        scale_lr_linear(base_learning_rate_at_reference_batch, eff, reference_effective_batch)
        if auto_scale_lr
        else float(current_lr if current_lr is not None else base_learning_rate_at_reference_batch)
    )
    accumulation_handled_by = "deepspeed" if mixed_precision_method == "deepspeed" else "manual"
    return LRScalingResult(
        num_gpus=int(max(1, num_gpus)),
        per_gpu_batch_size=int(max(1, per_gpu_batch_size)),
        gradient_accumulation_steps=int(max(1, gradient_accumulation_steps)),
        effective_batch_size=int(eff),
        reference_effective_batch=int(reference_effective_batch),
        base_lr_at_reference_batch=float(base_learning_rate_at_reference_batch),
        lr_actual=float(lr),
        backbone_coef_lr=float(backbone_coef_lr),
        backbone_lr=float(lr) * float(backbone_coef_lr),
        accumulation_handled_by=accumulation_handled_by,
    )


def apply_lr_scaling_to_args(args):
    result = compute_lr_scaling(
        per_gpu_batch_size=getattr(args, "per_gpu_train_batch_size", getattr(args, "train_batch_size", 1)),
        gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 1),
        reference_effective_batch=getattr(args, "reference_effective_batch", 64),
        base_learning_rate_at_reference_batch=getattr(args, "base_learning_rate_at_reference_batch", 2e-4),
        backbone_coef_lr=getattr(args, "backbone_coef_lr", 0.05),
        num_gpus=getattr(args, "num_gpus_for_lr", 1),
        auto_scale_lr=getattr(args, "auto_scale_lr", False),
        current_lr=getattr(args, "learning_rate", None),
        mixed_precision_method=getattr(args, "mixed_precision_method", ""),
    )
    args.effective_batch_size = result.effective_batch_size
    args.reference_effective_batch = result.reference_effective_batch
    args.base_learning_rate_at_reference_batch = result.base_lr_at_reference_batch
    args.lr_actual = result.lr_actual
    args.backbone_lr = result.backbone_lr
    args.loss_divided_by_accumulation = result.accumulation_handled_by == "manual"
    args.accumulation_handled_by = result.accumulation_handled_by
    if getattr(args, "auto_scale_lr", False):
        args.learning_rate = result.lr_actual
    return result
