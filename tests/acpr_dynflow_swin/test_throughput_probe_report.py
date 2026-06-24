from __future__ import annotations

from fate_x.engine.probe_acpr_dynflow_swin_throughput import (
    filter_candidate_configs,
    select_best_candidate,
    should_abort_after_warmup_memory,
)


def test_select_best_candidate_uses_samples_per_second_under_memory_and_time_gates():
    candidates = [
        {
            "batch_size": 8,
            "gradient_accumulation_steps": 8,
            "samples_per_second": 2.0,
            "projected_train_epoch_hours": 1.0,
            "peak_reserved_gib": 45.0,
            "data_time_fraction": 0.05,
            "finite": True,
            "optimizer_steps": 100,
        },
        {
            "batch_size": 4,
            "gradient_accumulation_steps": 16,
            "samples_per_second": 1.5,
            "projected_train_epoch_hours": 1.4,
            "peak_reserved_gib": 38.0,
            "data_time_fraction": 0.10,
            "finite": True,
            "optimizer_steps": 100,
        },
        {
            "batch_size": 2,
            "gradient_accumulation_steps": 32,
            "samples_per_second": 0.9,
            "projected_train_epoch_hours": 5.0,
            "peak_reserved_gib": 20.0,
            "data_time_fraction": 0.10,
            "finite": True,
            "optimizer_steps": 100,
        },
    ]
    selected = select_best_candidate(
        candidates,
        max_epoch_hours=4.0,
        max_peak_reserved_gib=44.0,
        max_data_time_fraction=0.20,
    )
    assert selected["batch_size"] == 4
    assert selected["selection_reason"] == "highest_samples_per_second_under_all_gates"


def test_warmup_memory_abort_only_when_hard_cap_is_exceeded():
    assert should_abort_after_warmup_memory(
        {"peak_reserved_gib": 45.0},
        max_peak_reserved_gib=44.0,
    ) is True
    assert should_abort_after_warmup_memory(
        {"peak_reserved_gib": 41.0},
        max_peak_reserved_gib=44.0,
    ) is False


def test_filter_candidate_configs_keeps_requested_batch_sizes():
    candidates = [
        {"batch_size": 8, "gradient_accumulation_steps": 8},
        {"batch_size": 4, "gradient_accumulation_steps": 16},
        {"batch_size": 2, "gradient_accumulation_steps": 32},
    ]
    filtered = filter_candidate_configs(candidates, "4,2")
    assert [item["batch_size"] for item in filtered] == [4, 2]
