from fate_x.utils.acpr_flow_config import load_acpr_flow_config
from fate_x.utils.acpr_flow_git_guard import _windows_gitdir_to_wsl


def test_config_contract_for_direct_image_no_cache_no_legacy():
    cfg = load_acpr_flow_config("configs/acpr_flowcal_pp_v1_bddx_32f_224.yaml")
    assert cfg["data"]["direct_image_training"] is True
    assert cfg["data"]["feature_cache_enabled"] is False
    assert cfg["data"]["token_cache_enabled"] is False
    assert cfg["data"]["build_cache_before_training"] is False
    assert cfg["evaluation"]["eval_splits"] == ["test"]


def test_git_guard_converts_windows_worktree_gitdir_to_wsl_path():
    assert _windows_gitdir_to_wsl("E:/sbw/ADAPT_repro/ADAPT/.git/worktrees/x") == "/mnt/e/sbw/ADAPT_repro/ADAPT/.git/worktrees/x"
    assert _windows_gitdir_to_wsl("C:\\Users\\WLJTXY\\repo\\.git") == "/mnt/c/Users/WLJTXY/repo/.git"
