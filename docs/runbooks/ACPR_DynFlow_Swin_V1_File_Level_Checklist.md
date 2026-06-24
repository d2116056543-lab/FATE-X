# ACPR-DynFlow-Swin V1 file-level implementation checklist

This checklist is subordinate to the complete plan and audit skill. Agent A and Agent B must map each row to code, tests, and a dynamic evidence artifact.

## A. New formal package

| File | Required public objects | Mandatory proof |
|---|---|---|
| `config.py` | typed config loader, unknown-key rejection, config-consumer manifest | mutated config changes runtime |
| `types.py` | all named dataclasses from plan | one real-batch shape report |
| `signal_codec.py` | train stats, encode/decode, official evaluator adapter | round-trip and ADAPT parity |
| `video_swin_backbone.py` | repository Video Swin-B wrapper, one-forward counter, native stages | BF16/native-stage proof |
| `predicate_ontology.py` | exact 32 names and priors | ontology hash/order |
| `predicate_transfer.py` | OIA mapper, true BERT name embedding, gate, residual | source/key/SHA report |
| `ego_motion.py` | bounded common-shift correlation | known-translation test |
| `dynamic_predicate_field.py` | recurrent entmax predicate field | order/region/gradient tests |
| `nnpu_calalign.py` | complete rules, nnPU, train-only calibration | real positive/reliable-negative counts |
| `semantic_token_consolidator.py` | five-slot mass-preserving consolidation | exact conservation/provenance |
| `predicate_covariates.py` | heterogeneous measurements to common representation | group ablations |
| `mesoscopic_corridor_flow.py` | distinct left/center/right flow | synthetic lane cases |
| `pattern_lag_traffic_reasoner.py` | dilation 1/2/4 patterns, 13 factors, lag 0–3 | connected path and lag recovery |
| `query_motion_transformer.py` | 12-layer BERT-capacity query motion model | target independence |
| `decision_ledger.py` | exact factor contributions, benefit gate, global/final output | exact sum and safe-loss direction |
| `contribution_reason_adapter.py` | action/explanation factor readers before LM head | image-hidden unchanged |
| `text_decoder.py` | ADAPT-compatible sep-cap training/generation | real autoregressive decode |
| `interventions.py` | earliest-layer re-forward engine | actual text/control deltas |
| `model.py` | typed orchestration only | no monolithic hidden bypass |

## B. Losses

`fate_x/losses/acpr_dynflow_swin_losses.py` must expose:

```text
signal_specific_normalized_huber
target_delta_loss
nnpu_loss
pattern_semantic_loss
traffic_state_semantic_loss
residual_target_loss
benefit_gate_loss
non_degradation_hinge
contribution_alignment_js
group_sparsity
temporal_smoothness
```

It must not expose the legacy magnitude-as-residual objective.

## C. Engines

| File | Required behavior |
|---|---|
| `acpr_dynflow_swin_data.py` | direct images, raw text, aligned IDs, no formal val loader |
| `eval_adapt_reference_dynflow.py` | comparison-only ADAPT evaluation and parity artifacts |
| `train_acpr_dynflow_swin.py` | one fixed 16-epoch run, BF16, param groups, per-epoch test |
| `eval_acpr_dynflow_swin.py` | exact ADAPT metrics plus intermediate outputs |
| `probe_acpr_dynflow_swin_throughput.py` | real 100-step component timing and memory |
| `run_acpr_dynflow_swin_preflight.py` | executes all blocking gates |
| `audit_acpr_dynflow_swin.py` | rejects missing/placeholder evidence |
| `export_acpr_dynflow_swin_visuals.py` | real Canvas |
| `build_acpr_dynflow_swin_atlas.py` | standalone HTML/JSON Atlas |
| `supervise_acpr_dynflow_swin_foreground.py` | attached full-suite supervisor |

## D. Shared files

| Existing file | Allowed change |
|---|---|
| `src/modeling/load_swin.py` | backward-compatible native-stage return API |
| `src/modeling/video_swin/swin_transformer.py` | native intermediate stage exposure only |
| `src/layers/bert/modeling_bert.py` | generic optional contribution adapter hook and generation state propagation |
| `src/datasets/vision_language_tsv.py` | raw action/justification/sample ID metadata |
| `src/datasets/vl_dataloader.py` | string-preserving collate and worker configuration |
| `src/datasets/caption_tensorizer.py` | ADAPT parity fixes only |
| `.gitignore` | ignore run artifacts/caches, not source |

## E. Formal import prohibitions

Formal code must not import:

```text
fate_x.acpr_dynflow
fate_x.acpr_flow
fate_x.acpr_flow_v2
fate_x.models.flowtrace_pmt_model
fate_x.models.token_pmt_adapter
fate_x.models.sinkhorn_transport
```

## F. Required implementation order

1. Read canonical records and snapshot current worktree.
2. Install plan/config/skill/manifest.
3. Create formal namespace and typed contracts.
4. Prove ADAPT metric parity and signal codec.
5. Implement repository Video Swin-B native-stage/BF16 wrapper.
6. Implement OIA transfer and predicate field.
7. Implement complete nnPU and CalAlign.
8. Implement mass-preserving consolidation.
9. Implement corridor flow and integrated pattern–lag reasoner.
10. Implement query motion transformer.
11. Implement exact decision ledger and losses.
12. Implement autoregressive decoder integration.
13. Implement trainer/optimizer/scheduler/BF16.
14. Implement evaluation/best selection.
15. Implement real interventions.
16. Implement Canvas/Atlas.
17. Implement real throughput probe.
18. Implement strict preflight/audit/supervisor.
19. Agent B review loop.
20. Formal foreground run.

No later item can be marked complete while an earlier contract test fails.
