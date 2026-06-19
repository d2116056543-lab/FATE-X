from fate_x.utils.acpr_flow_config import load_acpr_flow_config


def test_formal_protocol_uses_test_only_no_validation_loader():
    cfg = load_acpr_flow_config("configs/acpr_flowcal_pp_v1_bddx_32f_224.yaml")
    assert cfg["protocol_tag"] == "test_selected_user_requested"
    assert cfg["evaluation"]["eval_splits"] == ["test"]
    assert cfg["evaluation"]["best_selection_split"] == "test"
    assert cfg["evaluation"]["metric_based_early_stop"] is False
