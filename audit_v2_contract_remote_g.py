from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(r"E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree")
PKG = ROOT / "docs" / "runbooks"


REQUIRED_SYMBOLS = {
    "fate_x/acpr_flow_v2/config.py": [
        "FlowCalV2Config",
        "load_flowcal_v2_config",
        "write_resolved_config",
        "build_config_binding_manifest",
    ],
    "fate_x/acpr_flow_v2/types.py": [
        "FlowCalV2Batch",
        "VideoBackboneOutput",
        "LocalTransportOutput",
        "PredicateTrajectory",
        "LaneFlowFieldOutput",
        "AxisAwareFlowOutput",
        "SemanticReasonMemory",
        "FlowCalV2Bundle",
        "FlowCalV2TrainOutput",
        "GeneratedSequence",
        "InterventionSpecV2",
    ],
    "fate_x/acpr_flow_v2/adapt_video_backbone.py": [
        "ADAPTVideoBackboneV2",
        "_extract_native_stages",
        "_temporal_align",
        "_project_dense_tokens",
        "_fuse_reasoning_grids",
    ],
    "fate_x/acpr_flow_v2/adapt_motion_backbone.py": ["ADAPTMotionBackbone", "from_adapt_checkpoint", "predict"],
    "fate_x/acpr_flow_v2/local_partial_transport.py": [
        "LocalPartialTransportV2",
        "warp_source_map_to_current",
        "expected_transport_displacement",
    ],
    "fate_x/acpr_flow_v2/temporal_predicate_tracker.py": ["TransportedNamedPredicateTracker"],
    "fate_x/acpr_flow_v2/lane_flow_field.py": [
        "PredicateConditionedLaneFlowField",
        "build_soft_corridor_masks",
        "refine_masks_with_drivable_predicates",
        "aggregate_region_statistics",
        "temporal_encode_regions",
    ],
    "fate_x/acpr_flow_v2/axis_aware_flow_composer.py": ["AxisAwareFlowComposer", "derive_axis_direction_targets"],
    "fate_x/acpr_flow_v2/contextual_reason_target.py": ["FrozenContextualReasonTarget", "ActionSubspaceTracker"],
    "fate_x/acpr_flow_v2/pu_targets.py": ["FreeTextPUTargetBuilderV2", "positive_unlabeled_loss_v2"],
    "fate_x/acpr_flow_v2/semantic_reason_memory.py": [
        "SemanticReasonMemoryBuilder",
        "longitudinal_memory_mask",
        "lateral_memory_mask",
    ],
    "fate_x/acpr_flow_v2/semantic_gradient_firewall.py": ["scaled_gradient", "representation_pcgrad_surrogate"],
    "fate_x/acpr_flow_v2/temporal_seca.py": ["TemporalSECAV2"],
    "fate_x/acpr_flow_v2/axis_aware_control_adapter.py": ["AxisAwareReasonControlAdapter"],
    "fate_x/acpr_flow_v2/temporal_hardpair.py": ["ContradictionAwareTemporalHardPair"],
    "fate_x/acpr_flow_v2/prefix_future.py": ["PrefixFuturePredictor", "build_prefix_bundle_from_precomputed_grids"],
    "fate_x/acpr_flow_v2/sequence_calalign.py": ["SequenceCalAlignV2Scales", "SequenceCalAlignV2"],
    "fate_x/acpr_flow_v2/interventions.py": ["FlowCalV2InterventionEngine"],
    "fate_x/acpr_flow_v2/model.py": [
        "ACPRFlowCalV2Model",
        "build_visual_state",
        "build_reason_state",
        "forward_text",
        "forward_control",
        "decode_adapt_compatible",
        "generate_explanation_with_logprobs",
    ],
    "fate_x/losses/acpr_flowcal_v2_losses.py": [
        "shortest_circular_delta",
        "normalized_control_huber",
        "transport_consistency_loss",
        "lane_temporal_consistency_loss",
        "axis_direction_weak_loss",
        "delta_kl_loss",
        "parameter_anchor_loss",
        "memory_diversity_loss",
    ],
    "fate_x/losses/explanation_scst.py": [
        "sentence_cider_reward",
        "sentence_meteor_reward",
        "hallucination_penalty",
        "self_critical_explanation_loss",
    ],
    "fate_x/engine/acpr_flowcal_v2_data.py": [
        "resolve_adapt_text_contract",
        "build_v2_dataloader",
        "adapt_batch_to_v2",
        "stream_train_control_stats",
        "deterministic_train_calib_ids",
    ],
    "fate_x/engine/train_acpr_flowcal_v2.py": [
        "StageController",
        "StageAwareScheduler",
        "TestBestSelector",
        "CheckpointMigratorV1ToV2",
        "build_optimizer_groups",
        "train_one_epoch",
        "evaluate_after_epoch",
        "save_checkpoint_atomic",
        "load_resume_exact",
        "run_formal_suite",
    ],
    "fate_x/engine/probe_acpr_flowcal_v2_memory.py": [],
    "fate_x/engine/export_acpr_flowcal_v2_visuals.py": [],
    "fate_x/engine/build_acpr_flowcal_v2_atlas.py": [],
}

REQUIRED_TESTS = [
    "test_v2_adapt_motion_equivalence.py",
    "test_v2_motion_target_independence.py",
    "test_v2_seca_segment_readers.py",
    "test_v2_scst_logprob_reward.py",
    "test_v2_zero_gate_fallback.py",
    "test_v2_adapt_video_load.py",
    "test_v2_adapt_text_contract.py",
    "test_v2_test_only_protocol.py",
]

FORBIDDEN_SNIPPETS = [
    "from fate_x.acpr_flow.model",
    "import fate_x.acpr_flow.model",
    "TokenPMTAdapter",
    "FlowTraceLoss",
    "sinkhorn",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def symbols(path: Path) -> set[str]:
    if not path.exists():
        return set()
    tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(node.name)
    return out


def main() -> None:
    out_dir = Path(r"G:\\acpr_flowcal_v2_contract_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    missing_files = []
    missing_symbols = []
    for rel, required in REQUIRED_SYMBOLS.items():
        path = ROOT / rel
        present = path.exists()
        if not present:
            missing_files.append(rel)
        found = symbols(path)
        missing = [name for name in required if name not in found]
        if missing:
            missing_symbols.append({"file": rel, "missing": missing, "found": sorted(found)})
        rows.append({"file": rel, "present": present, "missing_symbols": missing})

    tests_present = sorted(p.name for p in (ROOT / "tests/acpr_flowcal_v2").glob("test_*.py"))
    missing_tests = [name for name in REQUIRED_TESTS if name not in tests_present]

    forbidden_hits = []
    for base in ["fate_x/acpr_flow_v2"]:
        for path in (ROOT / base).glob("**/*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for needle in FORBIDDEN_SNIPPETS:
                if needle in text:
                    forbidden_hits.append({"file": str(path.relative_to(ROOT)), "needle": needle})

    manifest = json.loads((PKG / "ACPR_FlowCal_V2_package_manifest.json").read_text(encoding="utf-8"))
    package_hashes = {}
    copy_map = {
        "Codex_ACPR_FlowCal_V2_Implementation_Plan.md": ROOT / "docs/runbooks/ACPR_FlowCal_V2_Implementation_Plan.md",
        "ACPR_FlowCal_V2_File_Level_Checklist.md": ROOT / "docs/runbooks/ACPR_FlowCal_V2_File_Level_Checklist.md",
        "Codex_ACPR_FlowCal_V2_Bootstrap_Prompt.txt": ROOT / "docs/runbooks/Codex_ACPR_FlowCal_V2_Bootstrap_Prompt.txt",
        "acpr-flowcal-v2-implementation-audit_SKILL.md": ROOT / ".codex/skills/acpr-flowcal-v2-implementation-audit/SKILL.md",
        "acpr_flowcal_v2_bddx_32f_224.yaml": ROOT / "configs/acpr_flowcal_v2_bddx_32f_224.yaml",
    }
    for name, dst in copy_map.items():
        expected = manifest["files"][name]["sha256"]
        actual = sha256(dst) if dst.exists() else None
        package_hashes[name] = {"expected": expected, "actual": actual, "match": actual == expected}

    config_text = (ROOT / "configs/acpr_flowcal_v2_bddx_32f_224.yaml").read_text(encoding="utf-8", errors="ignore")
    config_checks = {
        "direct_image_training_true": "direct_image_training: true" in config_text,
        "feature_cache_disabled": "feature_cache_enabled: false" in config_text,
        "token_cache_disabled": "token_cache_enabled: false" in config_text,
        "mask_prob_0_5": "mask_prob: 0.5" in config_text,
        "max_masked_tokens_45": "max_masked_tokens: 45" in config_text,
        "num_workers_4": "num_workers: 4" in config_text,
        "epochs_15": "epochs: 15" in config_text,
    }

    passed = (
        not missing_files
        and not missing_symbols
        and not missing_tests
        and not forbidden_hits
        and all(v["match"] for v in package_hashes.values())
        and all(config_checks.values())
    )
    report = {
        "passed": passed,
        "missing_files": missing_files,
        "missing_symbols": missing_symbols,
        "missing_tests": missing_tests,
        "forbidden_hits": forbidden_hits,
        "package_hashes": package_hashes,
        "config_checks": config_checks,
        "file_rows": rows,
        "tests_present": tests_present,
        "conclusion": "BLOCK_TRAINING_UNTIL_FULL_PLAN_COVERAGE" if not passed else "ALLOW_NEXT_REVIEW_GATE",
    }
    (out_dir / "contract_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# ACPR FlowCal V2 Contract Audit",
        "",
        f"passed: `{passed}`",
        f"conclusion: `{report['conclusion']}`",
        "",
        "## Missing Files",
        *(f"- `{x}`" for x in missing_files),
        "",
        "## Missing Symbols",
    ]
    for item in missing_symbols:
        lines.append(f"- `{item['file']}` missing: {', '.join(item['missing'])}")
    lines += [
        "",
        "## Missing Required Tests",
        *(f"- `{x}`" for x in missing_tests),
        "",
        "## Package Hash Checks",
    ]
    for name, item in package_hashes.items():
        lines.append(f"- `{name}` match={item['match']} expected={item['expected']} actual={item['actual']}")
    (out_dir / "contract_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
