from pathlib import Path

from fate_x.acpr_flow_v2.config import build_config_binding_manifest, load_flowcal_v2_config


def test_config_binding_manifest_uses_v2_yaml_defaults():
    cfg = load_flowcal_v2_config(Path("configs/acpr_flowcal_v2_bddx_32f_224.yaml"))
    manifest = build_config_binding_manifest(cfg)
    assert manifest
    assert cfg.epochs == 15
    assert cfg.text_contract["mask_prob"] == 0.5
    assert cfg.text_contract["max_masked_tokens"] == 45
