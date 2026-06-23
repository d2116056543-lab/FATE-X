import os

def test_trainer_writes_best_names():
    text=open('fate_x/engine/train_acpr_dynflow.py', encoding='utf-8').read()
    for name in ['checkpoint_best_text.pth','checkpoint_best_control.pth','checkpoint_best_joint.pth','checkpoint_best_test.pth','checkpoint_latest.pth']:
        assert name in text



def test_trainer_uses_config_eval_cap_by_default():
    text = open('fate_x/engine/train_acpr_dynflow.py', encoding='utf-8').read()
    assert 'max_eval_samples: int = -1' in text
    assert 'p.add_argument("--max_eval_samples", type=int, default=-1)' in text
    assert 'eval_cfg.get("best_checkpoint_cases"' in text
    assert 'viz_cfg.get("best_checkpoint_cases"' in text
    assert 'eval_cfg.get("lightweight_flow_audit_samples", -1)' in text
    assert '"eval_max_samples": eval_max_samples' in text
    assert 'max_samples=eval_max_samples' in text
