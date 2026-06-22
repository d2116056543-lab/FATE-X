from fate_x.acpr_dynflow.nnpu_calalign import phrase_labels
from fate_x.acpr_dynflow.predicate_ontology import EXACT_32_PREDICATES
import torch

def test_unknown_is_unlabeled():
    y=phrase_labels(['unrelated'], {}, EXACT_32_PREDICATES, torch.device('cpu'))
    assert y.reliable_negative.sum().item() == 0
    assert y.unlabeled.sum().item() == 32

