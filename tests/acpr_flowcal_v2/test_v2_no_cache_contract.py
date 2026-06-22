from fate_x.engine.audit_acpr_flowcal_v2 import run_static_contract_audit


def test_v2_audit_declares_direct_images_and_no_cache():
    report = run_static_contract_audit(".")
    assert report["direct_image_training"] is True
    assert report["feature_cache_enabled"] is False
    assert report["token_cache_enabled"] is False
