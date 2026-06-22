from __future__ import annotations

import torch

from fate_x.acpr_dynflow.config import load_dynflow_config
from fate_x.acpr_dynflow.model import ACPRDynFlowModel
from fate_x.engine.acpr_dynflow_data import SyntheticDynFlowDataset, collate_dynflow


def make_output(length: int = 1):
    cfg = load_dynflow_config("configs/acpr_dynflow_v1_bddx_32f_224.yaml")
    model = ACPRDynFlowModel(cfg)
    dataset = SyntheticDynFlowDataset(length=length)
    batch = collate_dynflow([dataset[i] for i in range(length)])
    out = model(batch)
    return model, batch, out

