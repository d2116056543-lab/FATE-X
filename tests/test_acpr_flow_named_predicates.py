import torch

from fate_x.acpr_flow.region_priors import ACPR_PREDICATE_NAMES, build_region_prior_grid


def test_acpr_flow_has_exact_32_named_predicates():
    assert ACPR_PREDICATE_NAMES == [
        "traffic_light_red",
        "traffic_light_green",
        "stop_sign_present",
        "front_vehicle_close",
        "front_vehicle_far",
        "pedestrian_front",
        "cyclist_front",
        "obstacle_front",
        "lane_left_available",
        "lane_right_available",
        "left_turn_region",
        "right_turn_region",
        "drivable_center",
        "drivable_left",
        "drivable_right",
        "road_clear",
        "road_crowded",
        "parked_vehicle_left",
        "parked_vehicle_right",
        "vehicle_left",
        "vehicle_right",
        "traffic_light_visible",
        "traffic_sign_visible",
        "crosswalk_region",
        "intersection_region",
        "merging_left_context",
        "merging_right_context",
        "ego_lane_centered",
        "low_front_visibility",
        "open_left_gap",
        "open_right_gap",
        "global_scene_context",
    ]


def test_acpr_flow_region_priors_are_named_and_normalized():
    priors = build_region_prior_grid(height=7, width=7)
    assert priors.shape == (32, 7, 7)
    assert (priors >= 0).all()
    assert (priors.sum(dim=(-1, -2)) - 1.0).abs().max().item() < 1e-5
    assert not torch.allclose(priors[3], priors[15])
