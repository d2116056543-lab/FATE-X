from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import os

import torch
from torch import nn

from fate_x.acpr_flow.model import ACPRFlowModel, ACPRFlowModelConfig
from fate_x.engine.train_acpr_flowcal_pp import parse_adapt_caption_eval_report


class _TinyBackbone(nn.Module):
    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        b, t, _, _, _ = frames.shape
        device = frames.device
        fine = torch.randn(b, t, 2, 2, 8, device=device)
        coarse = torch.randn(b, t, 1, 1, 8, device=device)
        fused = torch.randn(b, t, 2, 2, 8, device=device)
        dense = torch.randn(b, 6, 8, device=device)
        return {"fine_grid": fine, "coarse_grid": coarse, "fused_grid": fused, "dense_tokens": dense}


class _CaptureCaptioner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen: dict | None = None

    def forward(self, **kwargs):
        self.seen = kwargs
        batch = kwargs["img_feats"].shape[0]
        max_len = int(kwargs["max_length"])
        return (
            torch.zeros(batch, 1, max_len, dtype=torch.long, device=kwargs["img_feats"].device),
            torch.zeros(batch, 1, device=kwargs["img_feats"].device),
        )


def test_acpr_model_accepts_adapt_decode_inputs_and_injects_bundle() -> None:
    captioner = _CaptureCaptioner()
    model = ACPRFlowModel(
        ACPRFlowModelConfig(state_dim=8, text_hidden_dim=8, bert_img_feature_dim=6, formal_backbone=False),
        backbone=_TinyBackbone(),
        captioning_model=captioner,
    )
    batch = 2
    max_len = 10
    outputs = model(
        is_decode=True,
        input_ids=torch.zeros(batch, max_len, dtype=torch.long),
        attention_mask=torch.ones(batch, max_len, max_len),
        token_type_ids=torch.zeros(batch, max_len, dtype=torch.long),
        img_feats=torch.randn(batch, 32, 3, 224, 224),
        masked_pos=torch.ones(batch, max_len, dtype=torch.long),
        car_info=torch.zeros(batch, 32, 2),
        do_sample=False,
        bos_token_id=101,
        pad_token_id=0,
        eos_token_ids=[102],
        mask_token_id=103,
        add_od_labels=False,
        od_labels_start_posid=max_len,
        max_length=max_len,
        use_sep_cap=True,
        num_beams=1,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_return_sequences=1,
        num_keep_best=1,
    )
    assert outputs[0].shape == (batch, 1, max_len)
    assert captioner.seen is not None
    assert captioner.seen["is_decode"] is True
    assert captioner.seen["img_feats"].shape == (batch, 6, 6)
    assert captioner.seen["acpr_flow_bundle"] is not None
    assert captioner.seen["acpr_temporal_seca"] is model.temporal_seca
    assert captioner.seen["acpr_text_len"] == max_len


def test_parse_adapt_caption_eval_report_uses_cider_des_plus_exp(tmp_path: Path) -> None:
    base = tmp_path / "pred.BDDX.testing_32frames.beam3.max15.eval.json"
    base.write_text("{}", encoding="utf-8")
    base.with_name(base.name.replace("BDDX", "BDDX_des")).write_text('{"CIDEr": 2.5, "Bleu_4": 0.1}', encoding="utf-8")
    base.with_name(base.name.replace("BDDX", "BDDX_exp")).write_text('{"CIDEr": 1.25, "Bleu_4": 0.2}', encoding="utf-8")

    report = parse_adapt_caption_eval_report(base, split="test", test_yaml="datasets/BDDX/testing_32frames.yaml", use_sep_cap=True)

    assert report["metric_family"] == "adapt_coco_caption"
    assert report["metric_name"] == "CIDEr_des_plus_exp"
    assert report["metric_value"] == 3.75
    assert report["des_metrics"]["CIDEr"] == 2.5
    assert report["exp_metrics"]["CIDEr"] == 1.25


def test_temporal_seca_handles_decode_hidden_shorter_than_configured_text_len() -> None:
    from fate_x.acpr_flow.temporal_seca import TemporalSECA

    seca = TemporalSECA(hidden_dim=8)
    hidden = torch.randn(2, 2, 8)
    reason_memory = torch.randn(2, 5, 8)
    token_type_ids = torch.zeros(2, 30, dtype=torch.long)

    out, info = seca(hidden, reason_memory, token_type_ids=token_type_ids, text_len=30)

    assert out.shape == hidden.shape
    assert info["token_reason_attention"].shape[:2] == (2, 2)

def test_temporal_seca_pads_short_decode_token_type_ids() -> None:
    from fate_x.acpr_flow.temporal_seca import TemporalSECA

    seca = TemporalSECA(hidden_dim=8)
    hidden = torch.randn(2, 32, 8)
    reason_memory = torch.randn(2, 5, 8)
    token_type_ids = torch.zeros(2, 2, dtype=torch.long)

    out, info = seca(hidden, reason_memory, token_type_ids=token_type_ids, text_len=30)

    assert out.shape == hidden.shape
    assert info["token_reason_attention"].shape[:2] == (2, 30)


def test_temporal_seca_repeats_reason_memory_for_beam_search_batch() -> None:
    from fate_x.acpr_flow.temporal_seca import TemporalSECA

    seca = TemporalSECA(hidden_dim=8)
    hidden = torch.randn(12, 4, 8)
    reason_memory = torch.randn(4, 5, 8)
    token_type_ids = torch.zeros(4, 4, dtype=torch.long)

    out, info = seca(hidden, reason_memory, token_type_ids=token_type_ids, text_len=4)

    assert out.shape == hidden.shape
    assert info["token_reason_attention"].shape == (12, 4, 5)


def test_spice_java_compat_options_are_added_for_modern_java(monkeypatch) -> None:
    from fate_x.engine.train_acpr_flowcal_pp import ensure_spice_java_compat_options

    monkeypatch.setenv("JAVA_TOOL_OPTIONS", "-Xmx2G")
    monkeypatch.setattr(
        "fate_x.engine.train_acpr_flowcal_pp.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="", stderr='java version "17.0.2"\n'),
    )

    report = ensure_spice_java_compat_options(allow_java8_autodiscovery=False)

    opts = os.environ["JAVA_TOOL_OPTIONS"]
    assert report["java_major"] == 17
    assert "-Xmx2G" in opts
    assert "--add-opens=java.base/java.lang=ALL-UNNAMED" in opts
    assert "--add-opens=java.base/java.math=ALL-UNNAMED" in opts
    assert "--add-opens=java.base/java.util=ALL-UNNAMED" in opts
    assert "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED" in opts
    assert "--add-opens=java.base/java.net=ALL-UNNAMED" in opts
    assert "--add-opens=java.base/java.text=ALL-UNNAMED" in opts
    assert "--add-opens=java.base/java.io=ALL-UNNAMED" in opts
    assert "--add-opens=java.base/java.time=ALL-UNNAMED" in opts


def test_spice_java_compat_prefers_java8_and_removes_add_opens(tmp_path: Path, monkeypatch) -> None:
    from fate_x.engine.train_acpr_flowcal_pp import ensure_spice_java_compat_options

    java_home = tmp_path / "temurin8"
    bin_dir = java_home / "bin"
    bin_dir.mkdir(parents=True)
    java_bin = bin_dir / ("java.exe" if os.name == "nt" else "java")
    java_bin.write_text("fake", encoding="utf-8")
    monkeypatch.setenv("JAVA_TOOL_OPTIONS", "-Xmx2G --add-opens=java.base/java.lang=ALL-UNNAMED")
    monkeypatch.setenv("PATH", "ORIGINAL_PATH")
    monkeypatch.setattr(
        "fate_x.engine.train_acpr_flowcal_pp.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="", stderr='openjdk version "1.8.0_492"\n'),
    )

    report = ensure_spice_java_compat_options(preferred_java_home=str(java_home))

    assert report["java_major"] == 8
    assert os.environ["JAVA_HOME"] == str(java_home)
    assert os.environ["PATH"].split(os.pathsep)[0] == str(bin_dir)
    assert os.environ["JAVA_TOOL_OPTIONS"] == "-Xmx2G"


def test_clear_spice_runtime_cache_recreates_tmp_and_cache(tmp_path: Path) -> None:
    from fate_x.engine.train_acpr_flowcal_pp import clear_spice_runtime_cache

    spice_dir = tmp_path / "src" / "evalcap" / "coco_caption" / "pycocoevalcap" / "spice"
    (spice_dir / "tmp").mkdir(parents=True)
    (spice_dir / "cache").mkdir(parents=True)
    (spice_dir / "tmp" / "stale.txt").write_text("bad", encoding="utf-8")
    (spice_dir / "cache" / "stale.lmdb").write_text("bad", encoding="utf-8")
    keep = spice_dir / "spice-1.0.jar"
    keep.write_text("jar", encoding="utf-8")

    clear_spice_runtime_cache(tmp_path)

    assert (spice_dir / "tmp").is_dir()
    assert (spice_dir / "cache").is_dir()
    assert not (spice_dir / "tmp" / "stale.txt").exists()
    assert not (spice_dir / "cache" / "stale.lmdb").exists()
    assert keep.read_text(encoding="utf-8") == "jar"

def test_spice_disable_cache_omits_lmdb_cache_argument(monkeypatch) -> None:
    import json
    from src.evalcap.coco_caption.pycocoevalcap.spice.spice import Spice

    captured = {}

    def fake_check_call(cmd, cwd=None):
        captured["cmd"] = cmd
        out_path = cmd[cmd.index("-out") + 1]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([{
                "image_id": 1,
                "scores": {"All": {"f": 0.5}, "Objects": {"f": 0.25}},
            }], f)

    monkeypatch.setenv("SPICE_DISABLE_CACHE", "1")
    monkeypatch.setattr("src.evalcap.coco_caption.pycocoevalcap.spice.spice.subprocess.check_call", fake_check_call)

    score, scores = Spice().compute_score({1: ["a car stops"]}, {1: ["a car stops"]})

    assert score == 0.5
    assert scores[0]["All"]["f"] == 0.5
    assert "-cache" not in captured["cmd"]
    assert "-out" in captured["cmd"]
