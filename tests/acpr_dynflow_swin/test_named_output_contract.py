
def test_formal_output_contract_is_named_dataclasses_not_tuple_tail():
    from dataclasses import fields, is_dataclass

    from fate_x.acpr_dynflow_swin.types import (
        ACPRDynFlowSwinOutput,
        DynFlowSwinBatch,
        DynFlowSwinTextOutput,
        ExactDecisionLedger,
        MotionTransformerOutput,
        SwinBackboneOutput,
    )

    for cls in [
        DynFlowSwinBatch,
        SwinBackboneOutput,
        MotionTransformerOutput,
        ExactDecisionLedger,
        DynFlowSwinTextOutput,
        ACPRDynFlowSwinOutput,
    ]:
        assert is_dataclass(cls), f"{cls.__name__} must remain a dataclass contract"
        assert fields(cls), f"{cls.__name__} must expose named fields"

    output_names = {field.name for field in fields(ACPRDynFlowSwinOutput)}
    assert {
        "total_loss",
        "loss_components",
        "backbone",
        "predicates",
        "semantic_tokens",
        "traffic",
        "motion",
        "ledger",
        "text",
        "diagnostics",
    }.issubset(output_names)


def test_model_uses_backbone_native_final_time_for_semantic_tokens():
    import torch

    from fate_x.acpr_dynflow_swin.types import SwinBackboneOutput

    projection = torch.nn.Linear(64, 256)
    batch_size = 2
    native_final_steps = 16
    dense = torch.randn(batch_size, native_final_steps * 49, 64)
    bb = SwinBackboneOutput(
        predicate_grid=torch.empty(batch_size, 8, 4, 4, 64),
        final_grid=torch.empty(batch_size, native_final_steps, 7, 7, 64),
        temporal_global=torch.empty(batch_size, native_final_steps, 64),
        dense_final_tokens=dense,
        forward_count=1,
    )

    projected = projection(bb.dense_final_tokens)
    final_tokens = projected.reshape(batch_size, int(bb.final_grid.shape[1]), 49, -1)

    assert final_tokens.shape == (batch_size, 16, 49, 256)
    assert final_tokens.shape[-1] == projection.out_features


def test_model_global_projection_accepts_native_swin_final_dim():
    import torch

    projection = torch.nn.LazyLinear(768)
    temporal_global = torch.randn(2, 16, 1024)
    out = projection(temporal_global)

    assert out.shape == (2, 16, 768)
