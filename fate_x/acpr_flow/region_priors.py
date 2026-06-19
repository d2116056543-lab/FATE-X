from __future__ import annotations

import torch
from torch import Tensor


ACPR_PREDICATE_NAMES = [
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

FLOW_FACTOR_NAMES = [
    "clear_open_flow",
    "stable_following",
    "dense_following",
    "queue_congestion",
    "forming",
    "stable",
    "releasing",
    "oscillating",
    "traffic_signal",
    "lead_vehicle_group",
    "merge_lane_constraint",
    "turn_intersection",
    "vulnerable_obstacle_conflict",
]


def _gaussian(height: int, width: int, cx: float, cy: float, sx: float, sy: float) -> Tensor:
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, height),
        torch.linspace(-1, 1, width),
        indexing="ij",
    )
    g = torch.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2) / 2.0)
    return g / g.sum().clamp_min(1e-12)


def build_region_prior_grid(height: int, width: int, device=None, dtype=None) -> Tensor:
    centers = []
    for name in ACPR_PREDICATE_NAMES:
        if "left" in name:
            centers.append((-0.55, 0.15, 0.35, 0.65))
        elif "right" in name:
            centers.append((0.55, 0.15, 0.35, 0.65))
        elif "traffic_light" in name or "sign" in name:
            centers.append((0.0, -0.65, 0.55, 0.25))
        elif "front" in name or "obstacle" in name or "crosswalk" in name:
            centers.append((0.0, 0.05, 0.45, 0.45))
        elif "drivable" in name or "ego_lane" in name or "road_clear" in name:
            centers.append((0.0, 0.45, 0.55, 0.45))
        elif "global" in name or "visibility" in name or "crowded" in name:
            centers.append((0.0, 0.0, 1.2, 1.2))
        else:
            centers.append((0.0, 0.0, 0.7, 0.7))
    priors = torch.stack([_gaussian(height, width, *c) for c in centers], dim=0)
    return priors.to(device=device, dtype=dtype)


def build_factor_support() -> tuple[Tensor, Tensor]:
    support = torch.zeros(len(FLOW_FACTOR_NAMES), len(ACPR_PREDICATE_NAMES))
    contradiction = torch.zeros_like(support)
    name_to_idx = {n: i for i, n in enumerate(ACPR_PREDICATE_NAMES)}
    factor_to_idx = {n: i for i, n in enumerate(FLOW_FACTOR_NAMES)}

    def s(f: str, *preds: str) -> None:
        for p in preds:
            support[factor_to_idx[f], name_to_idx[p]] = 1.0

    def c(f: str, *preds: str) -> None:
        for p in preds:
            contradiction[factor_to_idx[f], name_to_idx[p]] = 1.0

    s("clear_open_flow", "road_clear", "front_vehicle_far", "open_left_gap", "open_right_gap")
    c("clear_open_flow", "road_crowded", "front_vehicle_close", "obstacle_front")
    s("stable_following", "front_vehicle_far", "ego_lane_centered")
    s("dense_following", "front_vehicle_close", "road_crowded")
    s("queue_congestion", "front_vehicle_close", "road_crowded", "obstacle_front")
    s("forming", "merging_left_context", "merging_right_context", "intersection_region")
    s("stable", "ego_lane_centered", "road_clear")
    s("releasing", "traffic_light_green", "open_left_gap", "open_right_gap")
    s("oscillating", "road_crowded", "front_vehicle_close")
    s("traffic_signal", "traffic_light_red", "traffic_light_green", "traffic_light_visible")
    s("lead_vehicle_group", "front_vehicle_close", "front_vehicle_far")
    s("merge_lane_constraint", "merging_left_context", "merging_right_context")
    s("turn_intersection", "left_turn_region", "right_turn_region", "intersection_region")
    s("vulnerable_obstacle_conflict", "pedestrian_front", "cyclist_front", "obstacle_front")
    return support, contradiction
