from fate_x.acpr_dynflow.predicate_ontology import EXACT_32_PREDICATES

def test_exact_32_predicates():
    assert len(EXACT_32_PREDICATES) == 32
    assert 'traffic_light_red' in EXACT_32_PREDICATES
    assert 'global_scene_context' == EXACT_32_PREDICATES[-1]

