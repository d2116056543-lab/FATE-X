import json
from pathlib import Path
import torch
from fate_x.acpr_flow.model import ACPRFlowModel
from fate_x.engine.train_acpr_flowcal_pp import (
    load_acpr_flow_config,
    load_formal_checkpoints,
    build_formal_captioning_model,
    build_model_config,
    load_adapt_visual_path_weights,
    load_resume_state,
    _import_bert_tokenizer,
    evaluate_acpr_control_metrics,
)

cfg_path = 'configs/acpr_flowcal_pp_v1_bddx_32f_224.yaml'
ckpt = r'/mnt/e/sbw/FATE_Drive/active_runs/acpr_linux_b4w4_resume_predcorrfix_20260621_034020/train/checkpoint_latest.pth'
out = Path(r'/mnt/e/sbw/FATE_Drive/active_runs/acpr_control_temporal_fix_smoke_20260621_0515')
out.mkdir(parents=True, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = load_acpr_flow_config(cfg_path)
load_formal_checkpoints(cfg)
captioning_model = build_formal_captioning_model(cfg, device)
model = ACPRFlowModel(build_model_config(cfg, load_pretrained_backbone=True), captioning_model=captioning_model).to(device)
load_adapt_visual_path_weights(model, cfg['paths']['adapt_checkpoint'])
resume = load_resume_state(ckpt, model, None, device=device)
tok_cls = _import_bert_tokenizer()
tok = tok_cls.from_pretrained(cfg['paths']['bert_dir'], do_lower_case=True)
report = evaluate_acpr_control_metrics(model, cfg, split='test', batch_size=4, tokenizer=tok, device=device, eval_output_dir=out / 'control_eval_smoke', max_eval_samples=64)
summary = {
    'resume': resume,
    'available': report.get('available'),
    'delta_stats': report.get('traffic_flow_audit', {}).get('delta_stats'),
    'sample_count': report.get('traffic_flow_audit', {}).get('sample_count'),
}
print('CONTROL_TEMPORAL_FIX_SMOKE ' + json.dumps(summary, sort_keys=True))
(out / 'control_report.json').write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')