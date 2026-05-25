from __future__ import annotations

from fate_x.explain.phrase_attribution import find_phrase_hits


def test_light_traffic_does_not_match_traffic_light():
    hits = find_phrase_hits("The car moves through light traffic.")
    assert all(h.concept != "traffic_light" for h in hits)


def test_traffic_light_red_matches_priority_concept():
    hits = find_phrase_hits("The traffic light is red and cars are stopped ahead.")
    concepts = [h.concept for h in hits]
    assert "traffic_light_red" in concepts
    assert "front_vehicle" in concepts


def test_people_crossing_maps_to_pedestrian():
    hits = find_phrase_hits("People crossing ahead require the vehicle to slow.")
    assert any(h.concept == "pedestrian" for h in hits)


def test_clear_traffic_reason_does_not_match_front_vehicle():
    hits = find_phrase_hits("Because traffic is clear, the car can continue.")
    concepts = [h.concept for h in hits]
    assert "clear_road" in concepts
    assert "front_vehicle" not in concepts
