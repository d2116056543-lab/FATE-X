import os

def test_trainer_writes_best_names():
    text=open('fate_x/engine/train_acpr_dynflow.py', encoding='utf-8').read()
    for name in ['checkpoint_best_text.pth','checkpoint_best_control.pth','checkpoint_best_joint.pth','checkpoint_best_test.pth','checkpoint_latest.pth']:
        assert name in text

