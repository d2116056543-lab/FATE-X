from pathlib import Path

model = Path('fate_x/acpr_flow/model.py')
text = model.read_text(encoding='utf-8')
if 'import math\n' not in text:
    text = text.replace('from dataclasses import dataclass, field\n', 'from dataclasses import dataclass, field\nimport math\n')
anchor = '    def _masked_token_loss(self, logits: Tensor, target_ids: Tensor | None) -> Tensor:\n'
helper = '''    def _control_hidden_sequence(self, reason_state: Tensor, steps: int | None = None) -> Tensor:\n        steps = int(steps or self.config.num_frames)\n        base = self.control_hidden(reason_state).unsqueeze(1).expand(reason_state.shape[0], steps, -1)\n        time = torch.linspace(0.0, 1.0, steps, device=base.device, dtype=base.dtype).view(1, steps, 1)\n        freq = torch.arange(1, base.shape[-1] + 1, device=base.device, dtype=base.dtype).view(1, 1, -1)\n        # Fixed temporal code keeps the continuous-control path from producing\n        # identical per-frame predictions before reason-memory adaptation.\n        temporal_code = torch.sin(time * freq * math.pi)\n        return base + 0.02 * temporal_code\n\n    def predict_control_from_bundle(self, bundle: ACPRFlowBundle, steps: int | None = None) -> dict[str, Tensor]:\n        control_hidden = self._control_hidden_sequence(bundle.global_reason_state, steps=steps)\n        control_base = self.control_base(control_hidden)\n        ctrl = self.reason_control_adapter(control_base, control_hidden, bundle.reason_memory)\n        bundle.control_reason_attention = ctrl["control_reason_attention"]\n        bundle.control_delta = ctrl["control_delta"]\n        return ctrl\n\n'''
if '_control_hidden_sequence' not in text:
    if anchor not in text:
        raise SystemExit('method anchor not found')
    text = text.replace(anchor, helper + anchor)
old = '''        control_hidden = self.control_hidden(bundle.global_reason_state).unsqueeze(1).expand(b, 32, -1)\n        control_base = self.control_base(control_hidden)\n        ctrl = self.reason_control_adapter(control_base, control_hidden, bundle.reason_memory)\n        bundle.control_reason_attention = ctrl["control_reason_attention"]\n        bundle.control_delta = ctrl["control_delta"]\n        control_final = ctrl["control_final_prediction"]\n'''
new = '''        control_steps = control_targets.shape[1] if control_targets is not None and control_targets.ndim >= 3 else self.config.num_frames\n        ctrl = self.predict_control_from_bundle(bundle, steps=control_steps)\n        control_base = ctrl["control_base_prediction"]\n        control_final = ctrl["control_final_prediction"]\n'''
if old in text:
    text = text.replace(old, new)
elif 'ctrl = self.predict_control_from_bundle(bundle, steps=control_steps)' not in text:
    raise SystemExit('forward control block not found')
model.write_text(text, encoding='utf-8')

adapter = Path('fate_x/acpr_flow/reason_control_adapter.py')
ad_text = adapter.read_text(encoding='utf-8')
ad_old = '        return {"control_final_prediction": final, "control_delta": delta * gate.view(1, 1, -1), "control_reason_attention": attn}\n'
ad_new = '        return {"control_base_prediction": base_prediction, "control_final_prediction": final, "control_delta": delta * gate.view(1, 1, -1), "control_reason_attention": attn}\n'
if ad_old in ad_text:
    ad_text = ad_text.replace(ad_old, ad_new)
elif '"control_base_prediction": base_prediction' not in ad_text:
    raise SystemExit('adapter return block not found')
adapter.write_text(ad_text, encoding='utf-8')

train = Path('fate_x/engine/train_acpr_flowcal_pp.py')
tr_text = train.read_text(encoding='utf-8')
tr_old = '''                grids = model.backbone(batch.frames)\n                bundle = model.build_bundle(batch.frames, precomputed_grids=grids)\n                hidden = model.control_hidden(bundle.global_reason_state).unsqueeze(1).expand(batch.frames.shape[0], 32, -1)\n                base = model.control_base(hidden)\n                ctrl = model.reason_control_adapter(base, hidden, bundle.reason_memory)\n                pred = ctrl["control_final_prediction"]\n'''
tr_new = '''                grids = model.backbone(batch.frames)\n                bundle = model.build_bundle(batch.frames, precomputed_grids=grids)\n                ctrl = model.predict_control_from_bundle(bundle, steps=batch.car_info.shape[-1])\n                pred = ctrl["control_final_prediction"]\n'''
if tr_old in tr_text:
    tr_text = tr_text.replace(tr_old, tr_new)
elif 'ctrl = model.predict_control_from_bundle(bundle, steps=batch.car_info.shape[-1])' not in tr_text:
    raise SystemExit('eval control block not found')
train.write_text(tr_text, encoding='utf-8')

test = Path('tests/test_acpr_flow_control_temporal_path.py')
test.write_text('''from __future__ import annotations\n\nfrom types import SimpleNamespace\n\nimport torch\n\nfrom fate_x.acpr_flow.model import ACPRFlowModel, ACPRFlowModelConfig\nfrom fate_x.engine.train_acpr_flowcal_pp import summarize_traffic_flow_audit\n\n\ndef test_control_path_has_temporal_variation_for_prediction_delta_audit() -> None:\n    model = ACPRFlowModel(ACPRFlowModelConfig(state_dim=8, text_hidden_dim=16, num_frames=6, formal_backbone=False))\n    state = torch.zeros(4, 16)\n    hidden = model._control_hidden_sequence(state, steps=6)\n    assert not torch.allclose(hidden[:, 0], hidden[:, -1])\n\n    bundle = SimpleNamespace(global_reason_state=state, reason_memory=torch.randn(4, 5, 16))\n    ctrl = model.predict_control_from_bundle(bundle, steps=6)\n    pred = ctrl["control_final_prediction"]\n    assert pred.shape == (4, 6, 2)\n    target = pred.clone()\n    target[:, -1, 1] = target[:, 0, 1] + torch.linspace(-1.0, 1.0, 4)\n    flow = torch.stack([torch.linspace(0.1, 0.9, 4), torch.linspace(0.9, 0.1, 4)], dim=1)\n    audit = summarize_traffic_flow_audit(flow, None, pred, target, flow_factor_names=["up", "down"], signal_names=["course", "speed"])\n    assert audit["delta_stats"]["speed"]["pred_delta_zero_variance"] is False\n    assert audit["flow_factors"]["up"]["pred_speed_delta_corr_reason"] == "ok"\n''', encoding='utf-8')
print('patched control temporal path')
