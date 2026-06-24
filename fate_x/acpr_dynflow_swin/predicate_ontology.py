from __future__ import annotations

EXACT_32_PREDICATES: tuple[str, ...] = (
    "traffic_light_red",
    "traffic_light_yellow",
    "traffic_light_green",
    "stop_sign",
    "lead_vehicle_present",
    "lead_vehicle_braking",
    "lead_vehicle_moving",
    "vehicle_cut_in",
    "vehicle_on_left",
    "vehicle_on_right",
    "left_lane_open",
    "left_lane_blocked",
    "right_lane_open",
    "right_lane_blocked",
    "center_lane_clear",
    "center_lane_blocked",
    "queue_ahead",
    "dense_traffic",
    "open_road",
    "intersection",
    "turn_left_context",
    "turn_right_context",
    "merge_lane",
    "construction_or_barrier",
    "pedestrian_ahead",
    "cyclist_ahead",
    "vulnerable_road_user",
    "crosswalk",
    "rain_or_low_visibility",
    "road_curve",
    "ego_lane_marking",
    "unknown_or_other_risk",
)

if len(EXACT_32_PREDICATES) != 32:
    raise RuntimeError("ACPR-DynFlow-Swin requires exactly 32 predicates")


TRAFFIC_FACTOR_NAMES: tuple[str, ...] = (
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
)
