from pathlib import Path


def test_config_forbids_cache_and_val():
    text = Path("configs/flowtrace_pmt_v1_bddx_32f_224.yaml").read_text()
    assert "feature_cache_enabled: false" in text
    assert "token_cache_enabled: false" in text
    assert "eval_splits: [test]" in text
    assert "no_metric_early_stop: true" in text
