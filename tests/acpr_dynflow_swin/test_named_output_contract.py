
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
