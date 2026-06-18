from pathlib import Path
from types import SimpleNamespace

from fate_x.engine.flowtrace_adapt_bridge import build_adapt_train_command
from src.datasets.vl_dataloader import _resolve_limited_samples


def test_config_forbids_cache_and_val():
    text = Path("configs/flowtrace_pmt_v1_bddx_32f_224.yaml").read_text()
    assert "feature_cache_enabled: false" in text
    assert "token_cache_enabled: false" in text
    assert "eval_splits: [test]" in text
    assert "no_metric_early_stop: true" in text


def test_flowtrace_bridge_preserves_official_adapt_sparse_attention_contract(tmp_path):
    command = build_adapt_train_command(
        "configs/flowtrace_pmt_v1_bddx_32f_224.yaml",
        tmp_path,
        epochs=1,
        micro_batch=1,
        grad_accum=1,
        max_steps=1,
        beam_size=1,
    ).command

    assert "--learn_mask_enabled" in command
    assert "false" not in command[command.index("--learn_mask_enabled") + 1: command.index("--learn_mask_enabled") + 2]
    assert command[command.index("--attn_mask_type") + 1] == "learn_vid_att"
    assert command[command.index("--loss_sparse_w") + 1] == "0.1"


def test_flowtrace_bridge_preserves_official_adapt_deepspeed_fp16_contract(tmp_path):
    command = build_adapt_train_command(
        "configs/flowtrace_pmt_v1_bddx_32f_224.yaml",
        tmp_path,
        epochs=1,
        micro_batch=1,
        grad_accum=1,
        max_steps=1,
        beam_size=1,
    ).command

    assert command[command.index("--mixed_precision_method") + 1] == "deepspeed"
    assert command[command.index("--deepspeed_fp16") + 1] == "true"
    assert command[command.index("--deepspeed_bf16") + 1] == "false"


def test_flowtrace_bridge_bounds_real_smoke_eval_samples_only_when_requested(tmp_path):
    formal_command = build_adapt_train_command(
        "configs/flowtrace_pmt_v1_bddx_32f_224.yaml",
        tmp_path / "formal",
        epochs=40,
        micro_batch=1,
        grad_accum=1,
    ).command
    smoke_command = build_adapt_train_command(
        "configs/flowtrace_pmt_v1_bddx_32f_224.yaml",
        tmp_path / "smoke",
        epochs=1,
        micro_batch=1,
        grad_accum=1,
        max_steps=8,
        max_eval_samples=8,
        beam_size=1,
    ).command

    assert "--limited_eval_samples" not in formal_command
    assert smoke_command[smoke_command.index("--limited_eval_samples") + 1] == "8"


def test_flowtrace_bridge_adds_hard_train_step_limit_for_real_smoke(tmp_path):
    command = build_adapt_train_command(
        "configs/flowtrace_pmt_v1_bddx_32f_224.yaml",
        tmp_path / "smoke",
        epochs=1,
        micro_batch=1,
        grad_accum=1,
        max_steps=8,
        max_eval_samples=8,
        beam_size=1,
    ).command

    assert command[command.index("--flowtrace_max_train_steps") + 1] == "8"


def test_eval_sample_limit_is_independent_from_train_limit_and_single_gpu_safe():
    args = SimpleNamespace(limited_samples=8, limited_eval_samples=8, num_gpus=1)

    assert _resolve_limited_samples(args, is_train=True, fallback_num_gpus=8) == 8
    assert _resolve_limited_samples(args, is_train=False, fallback_num_gpus=8) == 8
