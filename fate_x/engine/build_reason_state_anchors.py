from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from fate_x.engine.flowtrace_adapt_bridge import _load_yaml
from fate_x.models.reason_state_anchors import ReasonAnchorArtifacts, ridge_residualize, save_anchor_artifacts, spherical_kmeans


def _read_train_texts(config_path: Path, limit: int | None = None) -> tuple[list[str], list[str], list[str]]:
    cfg = _load_yaml(config_path)
    train_yaml = Path(cfg["paths"]["train_yaml"])
    train_cfg = _load_yaml(train_yaml)
    caption_path = train_yaml.parent / train_cfg["caption"]
    sample_ids, actions, justifications = [], [], []
    with caption_path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(sample_ids) >= limit:
                break
            key, raw = line.rstrip("\n").split("\t", 1)
            rows = json.loads(raw)
            if not rows:
                continue
            row = rows[0]
            action = str(row.get("action", "")).strip()
            justification = str(row.get("justification", "")).strip()
            if not action or not justification:
                continue
            sample_ids.append(key)
            actions.append(action)
            justifications.append(justification)
    if not sample_ids:
        raise RuntimeError(f"No action/justification rows found in {caption_path}")
    return sample_ids, actions, justifications


def _hash_embed(texts: list[str], dim: int = 256) -> torch.Tensor:
    vectors = torch.zeros(len(texts), dim)
    for row, text in enumerate(texts):
        for token in text.lower().replace(".", " ").replace(",", " ").split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vectors[row, idx] += sign
    return F.normalize(vectors, dim=-1)


def _bert_embed(texts: list[str], bert_dir: str, batch_size: int = 32) -> torch.Tensor:
    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception:
        return _hash_embed(texts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(bert_dir, local_files_only=True)
    model = AutoModel.from_pretrained(bert_dir, local_files_only=True).to(device)
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            outputs.append(pooled.cpu())
    return F.normalize(torch.cat(outputs, dim=0), dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flowtrace_pmt_v1_bddx_32f_224.yaml")
    parser.add_argument("--output", default=".background_runs/flowtrace_reason_state_anchors/anchors.pt")
    parser.add_argument("--num_anchors", type=int, default=8)
    parser.add_argument("--max_train_texts", type=int, default=0)
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config))
    limit = None if args.max_train_texts <= 0 else args.max_train_texts
    sample_ids, actions, justifications = _read_train_texts(Path(args.config), limit=limit)
    action_emb = _bert_embed(actions, cfg["paths"]["bert_dir"])
    reason_emb = _bert_embed(justifications, cfg["paths"]["bert_dir"])
    residual = ridge_residualize(action_emb, reason_emb)
    anchors, labels = spherical_kmeans(residual, k=args.num_anchors)
    nearest = []
    for cluster in range(args.num_anchors):
        idxs = (labels == cluster).nonzero(as_tuple=False).flatten().tolist()
        if not idxs:
            nearest.append([])
            continue
        sims = residual[idxs] @ anchors[cluster]
        order = sims.argsort(descending=True)[:5].tolist()
        nearest.append([justifications[idxs[i]] for i in order])
    artifacts = ReasonAnchorArtifacts(
        anchors=anchors,
        train_sample_ids=sample_ids,
        nearest_justifications=nearest,
        fingerprint={
            "builder": "flowtrace_pmt_v1_train_caption_tsv",
            "train_only": True,
            "num_texts": len(sample_ids),
            "config": str(args.config),
            "bert_dir": cfg["paths"]["bert_dir"],
        },
    )
    save_anchor_artifacts(args.output, artifacts)
    print(f"saved_reason_anchors={args.output} num_texts={len(sample_ids)}")


if __name__ == "__main__":
    main()
