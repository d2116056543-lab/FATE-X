import json

import torch

from src.tasks.run_adapt import (
    _collect_flowtrace_grad_norms,
    _flowtrace_train_step_limit,
    _reached_flowtrace_train_step_limit,
    _stack_signal_rows,
    _update_flowtrace_smoke_evidence,
)


class TinyFlowTraceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flowtrace_encoder = torch.nn.Module()
        self.flowtrace_encoder.transport = torch.nn.Linear(2, 2)
        self.flowtrace_encoder.tracks = torch.nn.Module()
        self.flowtrace_encoder.tracks.track_queries = torch.nn.Parameter(torch.ones(2, 2))
        self.flowtrace_encoder.composer = torch.nn.Linear(2, 2)
        self.flowtrace_encoder.reason = torch.nn.Linear(2, 2)
        self.token_pmt_adapter = torch.nn.Linear(2, 2)

    def forward(self, x):
        y = self.flowtrace_encoder.transport(x)
        y = y + x @ self.flowtrace_encoder.tracks.track_queries
        y = self.flowtrace_encoder.composer(y)
        y = self.flowtrace_encoder.reason(y)
        return self.token_pmt_adapter(y).sum()


def test_collect_flowtrace_grad_norms_tracks_required_modules():
    model = TinyFlowTraceModule()
    loss = model(torch.ones(2, 2))
    loss.backward()

    norms = _collect_flowtrace_grad_norms(model)

    for key in ["transport", "track_queries", "state_composer", "reason_state_head", "pmt"]:
        assert norms[key] > 0.0


def test_update_flowtrace_smoke_evidence_merges_json(tmp_path):
    path = tmp_path / "summary.json"

    _update_flowtrace_smoke_evidence(path, {"real_data_smoke": True, "train_samples": 8})
    _update_flowtrace_smoke_evidence(path, {"eval_samples": 8})

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["real_data_smoke"] is True
    assert data["train_samples"] == 8
    assert data["eval_samples"] == 8


def test_stack_signal_rows_returns_none_for_empty_limited_eval():
    assert _stack_signal_rows([], []) is None


def test_flowtrace_smoke_train_step_limit_is_explicit_and_hard():
    args = type("Args", (), {"flowtrace_max_train_steps": "8"})()

    assert _flowtrace_train_step_limit(args) == 8
    assert _reached_flowtrace_train_step_limit(args, 7) is False
    assert _reached_flowtrace_train_step_limit(args, 8) is True
