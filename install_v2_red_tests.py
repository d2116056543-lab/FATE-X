from pathlib import Path

ROOT = Path(r"E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree")

files = {
    "tests/acpr_flowcal_v2/test_v2_sensor_pred_head_contract.py": r'''
from types import SimpleNamespace

import torch

from src.modeling.load_sensor_pred_head import Sensor_Pred_Head


def _args():
    return SimpleNamespace(
        img_feature_dim=8,
        grid_feat=True,
        config_name="",
        model_name_or_path="models/bert-base-uncased",
        signal_types=["course", "speed"],
    )


def test_sensor_pred_head_exposes_target_independent_encode_predict():
    model = Sensor_Pred_Head(_args())
    feats = torch.randn(2, 5, 8)

    hidden = model.encode(feats)
    assert hidden.shape[:2] == (2, 5)
    assert hidden.shape[-1] == model.config.hidden_size

    pred = model.predict(feats, frame_num=3)
    assert pred.shape == (2, 3, 2)


def test_sensor_pred_head_forward_remains_backward_compatible():
    model = Sensor_Pred_Head(_args())
    feats = torch.randn(2, 5, 8)
    target = torch.randn(2, 2, 5)

    loss, pred, hidden = model(img_feats=feats, car_info=target, return_hidden=True)
    assert loss.ndim == 0
    assert pred.shape == (2, 5, 2)
    assert hidden.shape[:2] == (2, 5)
''',
    "tests/acpr_flowcal_v2/test_v2_model_contract.py": r'''
import torch

from fate_x.acpr_flow_v2.config import ACPRFlowCalV2Config
from fate_x.acpr_flow_v2.model import ACPRFlowCalV2Model
from fate_x.acpr_flow_v2.types import FlowCalV2Batch


def test_v2_model_consumes_direct_frames_and_returns_required_outputs():
    cfg = ACPRFlowCalV2Config(hidden_dim=32, text_vocab_size=17, num_frames=32)
    model = ACPRFlowCalV2Model(cfg)
    batch = FlowCalV2Batch(
        frames=torch.randn(2, 32, 3, 224, 224),
        input_ids=torch.randint(0, 17, (2, 30)),
        attention_mask=torch.ones(2, 30, dtype=torch.long),
        masked_pos=torch.tensor([[1, 2], [3, 4]]),
        masked_ids=torch.randint(0, 17, (2, 2)),
        car_info=torch.randn(2, 2, 32),
        sample_ids=["a", "b"],
    )

    out = model(batch, stage="R")
    assert out.total_loss.ndim == 0
    assert out.text_logits.shape[:2] == (2, 30)
    assert out.control_pred.shape == (2, 32, 2)
    assert out.bundle.local_transport_probs.shape[0] == 2
    assert "traffic_density" in out.bundle.diagnostics
''',
    "tests/acpr_flowcal_v2/test_v2_losses_and_eval.py": r'''
import torch

from fate_x.losses.acpr_flowcal_v2_losses import masked_language_model_loss, control_rmse_loss
from fate_x.engine.evaluate_v51_event_metrics import compute_text_cider_proxy


def test_v2_losses_are_finite_and_differentiable():
    logits = torch.randn(2, 5, 11, requires_grad=True)
    labels = torch.randint(0, 11, (2, 3))
    pos = torch.tensor([[0, 2, 4], [1, 3, 4]])
    mlm = masked_language_model_loss(logits, labels, pos)
    pred = torch.randn(2, 32, 2, requires_grad=True)
    target = torch.randn(2, 2, 32)
    ctrl = control_rmse_loss(pred, target)
    loss = mlm + ctrl
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert pred.grad is not None


def test_cider_proxy_prefers_exact_text():
    exact = compute_text_cider_proxy(["car slows down"], ["car slows down"])
    wrong = compute_text_cider_proxy(["car speeds up"], ["car slows down"])
    assert exact > wrong
''',
    "tests/acpr_flowcal_v2/test_v2_audit_static_contract.py": r'''
from pathlib import Path

from fate_x.engine.audit_acpr_flowcal_v2 import run_static_contract_audit


def test_static_contract_rejects_legacy_imports(tmp_path):
    root = tmp_path / "repo"
    (root / "fate_x/acpr_flow_v2").mkdir(parents=True)
    (root / "fate_x/acpr_flow_v2/model.py").write_text(
        "from fate_x.acpr_flow.model import ACPRFlowModel\n", encoding="utf-8"
    )
    report = run_static_contract_audit(root)
    assert report["forbidden_imports"], report


def test_static_contract_accepts_v2_namespace_without_legacy_imports(tmp_path):
    root = tmp_path / "repo"
    (root / "fate_x/acpr_flow_v2").mkdir(parents=True)
    (root / "fate_x/acpr_flow_v2/model.py").write_text("import torch\n", encoding="utf-8")
    report = run_static_contract_audit(root)
    assert not report["forbidden_imports"], report
''',
}

for rel, text in files.items():
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip(), encoding="utf-8")

print(f"wrote {len(files)} V2 red tests")
