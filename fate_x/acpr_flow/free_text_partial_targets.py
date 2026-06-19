from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import Tensor

from .region_priors import ACPR_PREDICATE_NAMES, FLOW_FACTOR_NAMES


@dataclass(frozen=True)
class TextRule:
    pattern: str
    positive: tuple[str, ...] = ()
    contradictory: tuple[str, ...] = ()


DEFAULT_TEXT_RULES = [
    TextRule(r"traffic (is )?clear|road (is )?clear|open road", ("road_clear", "clear_open_flow"), ("road_crowded", "queue_congestion")),
    TextRule(r"cars? ahead|vehicle ahead|lead vehicle", ("front_vehicle_close", "lead_vehicle_group"), ("front_vehicle_far",)),
    TextRule(r"stopped|queue|congestion|crowded", ("queue_congestion", "road_crowded"), ("clear_open_flow", "road_clear")),
    TextRule(r"pedestrian|person|crosswalk", ("pedestrian_front", "crosswalk_region", "vulnerable_obstacle_conflict"), ()),
    TextRule(r"cyclist|bike|bicycle", ("cyclist_front", "vulnerable_obstacle_conflict"), ()),
    TextRule(r"red light", ("traffic_light_red", "traffic_signal"), ("traffic_light_green",)),
    TextRule(r"green light", ("traffic_light_green", "traffic_signal", "releasing"), ("traffic_light_red",)),
    TextRule(r"left turn", ("left_turn_region", "turn_intersection"), ()),
    TextRule(r"right turn", ("right_turn_region", "turn_intersection"), ()),
]


class FreeTextPartialTargetBuilder:
    def __init__(self, unknown_negative_weight: float = 0.075, rules: list[TextRule] | None = None) -> None:
        self.unknown_negative_weight = float(unknown_negative_weight)
        self.rules = rules or DEFAULT_TEXT_RULES
        self.pred_index = {n: i for i, n in enumerate(ACPR_PREDICATE_NAMES)}
        self.flow_index = {n: i for i, n in enumerate(FLOW_FACTOR_NAMES)}

    def build(self, raw_actions: list[str], raw_justifications: list[str], device=None) -> dict[str, Tensor]:
        b = len(raw_actions)
        pred_pos = torch.zeros(b, len(ACPR_PREDICATE_NAMES), device=device)
        pred_con = torch.zeros_like(pred_pos)
        pred_mask = torch.zeros_like(pred_pos)
        flow_pos = torch.zeros(b, len(FLOW_FACTOR_NAMES), device=device)
        flow_con = torch.zeros_like(flow_pos)
        flow_mask = torch.zeros_like(flow_pos)
        for i, (a, j) in enumerate(zip(raw_actions, raw_justifications)):
            text = f"{a or ''} {j or ''}".lower()
            for rule in self.rules:
                if not re.search(rule.pattern, text):
                    continue
                for name in rule.positive:
                    if name in self.pred_index:
                        pred_pos[i, self.pred_index[name]] = 1
                        pred_mask[i, self.pred_index[name]] = 1
                    if name in self.flow_index:
                        flow_pos[i, self.flow_index[name]] = 1
                        flow_mask[i, self.flow_index[name]] = 1
                for name in rule.contradictory:
                    if name in self.pred_index:
                        pred_con[i, self.pred_index[name]] = 1
                        pred_mask[i, self.pred_index[name]] = 1
                    if name in self.flow_index:
                        flow_con[i, self.flow_index[name]] = 1
                        flow_mask[i, self.flow_index[name]] = 1
        return {
            "predicate_positive": pred_pos,
            "predicate_contradiction": pred_con,
            "predicate_known_mask": pred_mask,
            "predicate_reliability": torch.where(pred_mask > 0, torch.ones_like(pred_mask), torch.full_like(pred_mask, self.unknown_negative_weight)),
            "flow_positive": flow_pos,
            "flow_contradiction": flow_con,
            "flow_known_mask": flow_mask,
            "flow_reliability": torch.where(flow_mask > 0, torch.ones_like(flow_mask), torch.full_like(flow_mask, self.unknown_negative_weight)),
        }
