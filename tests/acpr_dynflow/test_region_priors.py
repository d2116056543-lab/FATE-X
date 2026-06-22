from fate_x.acpr_dynflow.predicate_ontology import region_prior_name

def test_region_prior_names():
    assert region_prior_name('lane_left_available') == 'left'
    assert region_prior_name('front_vehicle_close') == 'center'

