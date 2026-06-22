from __future__ import annotations

import hashlib
from dataclasses import dataclass

EXACT_32_PREDICATES: tuple[str, ...] = (
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
)

PATTERN_NAMES: tuple[str, ...] = ("stable", "forming", "releasing", "oscillating")

TRAFFIC_FACTOR_NAMES: tuple[str, ...] = (
    "clear_open_flow",
    "stable_following",
    "dense_following",
    "queue_congestion",
    "traffic_signal",
    "lead_vehicle_group",
    "merge_lane_constraint",
    "turn_intersection",
    "vulnerable_obstacle_conflict",
    "left_gap_opening",
    "right_gap_opening",
    "visibility_limited",
    "global_context",
)


@dataclass(frozen=True)
class PredicateOntology:
    names: tuple[str, ...] = EXACT_32_PREDICATES

    @property
    def sha256(self) -> str:
        return hashlib.sha256("\n".join(self.names).encode("utf-8")).hexdigest()


def region_prior_name(predicate: str) -> str:
    if "left" in predicate:
        return "left"
    if "right" in predicate:
        return "right"
    if "front" in predicate or "center" in predicate or "clear" in predicate:
        return "center"
    return "global"

