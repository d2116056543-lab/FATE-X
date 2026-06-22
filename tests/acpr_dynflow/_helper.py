from __future__ import annotations

import torch

from fate_x.acpr_dynflow.config import load_dynflow_config
from fate_x.acpr_dynflow.model import ACPRDynFlowModel
from fate_x.engine.acpr_dynflow_data import SyntheticDynFlowDataset, collate_dynflow


def make_output():
    cfg = load_dynflow_config("configs/acpr_dynflow_v1_bddx_32f_224.yaml")
    model = ACPRDynFlowModel(cfg)
    batch = collate_dynflow([SyntheticDynFlowDataset(length=1)[0]])
    out = model(batch)
    return model, batch, out

