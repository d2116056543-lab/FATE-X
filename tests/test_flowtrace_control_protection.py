from pathlib import Path


def test_control_branch_keeps_dense_tokens():
    text = Path("src/modeling/multitask_e2e_vid_swin_bert.py").read_text()
    assert "control_kwargs['img_feats'] = vid_feats_control" in text
    assert "text_kwargs['flowtrace_bundle']" in text
    assert "control_kwargs['flowtrace_bundle']" not in text
