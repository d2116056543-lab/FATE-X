from __future__ import annotations


def validate_fate_x_mask_compatibility(args) -> None:
    """Reject unsafe ADAPT sparse-mask settings with compressed FATE-X tokens.

    ADAPT's learned video sparse mask is parameterized at the original
    max_img_seq_length. FATE-X token compression changes the visual token
    count before the VL transformer, so the original learned mask can no
    longer be written into the compressed attention block without a separate
    compressed-mask implementation.
    """
    fate_enabled = bool(getattr(args, "fate_x_enabled", False))
    reducer = str(getattr(args, "video_token_reducer", "none"))
    learn_mask = bool(getattr(args, "learn_mask_enabled", False))
    if fate_enabled and reducer != "none" and learn_mask:
        raise ValueError(
            "FATE-X token compression is incompatible with ADAPT "
            "learn_mask_enabled. Run FATE-X with --learn_mask_enabled false "
            "and --loss_sparse_w 0, or implement a compressed learned mask."
        )
    reduce_control = bool(getattr(args, "fate_x_reduce_control", False))
    control_reducer = str(getattr(args, "fate_x_control_reducer", "none"))
    if fate_enabled and reduce_control and control_reducer not in {"temporal_ordered_topk"}:
        raise ValueError(
            "FATE-X control branch reduction must be temporal-order preserving. "
            "Use --fate_x_reduce_control false for the default dense CSP path, "
            "or set --fate_x_control_reducer temporal_ordered_topk explicitly."
        )
