from dataclasses import fields, is_dataclass

from fate_x.acpr_flow_v2.types import FlowCalV2TrainOutput, SemanticReasonMemory


def test_named_dataclass_outputs_expose_required_fields():
    assert is_dataclass(FlowCalV2TrainOutput)
    names = {f.name for f in fields(FlowCalV2TrainOutput)}
    assert {"action_text_loss", "explanation_text_loss", "speed_loss", "course_loss", "bundle"} <= names
    memory_names = {f.name for f in fields(SemanticReasonMemory)}
    assert {"values", "mask", "confidence", "names", "type_ids", "axis_ids", "lineage"} <= memory_names
