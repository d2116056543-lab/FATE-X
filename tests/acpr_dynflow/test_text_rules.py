from fate_x.acpr_dynflow.nnpu_calalign import phrase_labels
from fate_x.acpr_dynflow.predicate_ontology import EXACT_32_PREDICATES
import torch

def test_light_traffic_exclusion():
    labels=phrase_labels(['light traffic ahead'], {'traffic_light_red': {'positive':['light'], 'contradiction':[], 'exclusion':['light traffic']}}, EXACT_32_PREDICATES, torch.device('cpu'))
    assert labels.positive.sum().item() == 0

