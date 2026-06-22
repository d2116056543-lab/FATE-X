# ACPR-DynFlow V1 file-level implementation checklist

This checklist supplements the full implementation plan. Agent A and Agent B must mark every row with code path, test path, and evidence artifact.

## New formal package

| File | Required public objects | Dynamic evidence |
|---|---|---|
| `fate_x/acpr_dynflow/config.py` | typed config loader, unknown-key rejection, consumer manifest | config-binding report |
| `types.py` | all named dataclasses in plan | one real-batch tensor contract |
| `signal_codec.py` | train stats, encode/decode, official metric adapter | round-trip and ADAPT parity |
| `video_backbone.py` | independent Kinetics Video Swin wrapper | one-forward counter, stage gradients |
| `predicate_ontology.py` | exact 32 names, region priors, grammar metadata | ontology hash |
| `predicate_transfer.py` | OIA checkpoint mapper + BERT name init + residual | loaded-key and contribution report |
| `ego_motion.py` | coarse common-shift estimator | known-translation test |
| `dynamic_predicate_field.py` | recurrent query, entmax evidence, confidence/centroid/lane mass | order sensitivity and gradients |
| `nnpu_calalign.py` | rules, nnPU, train-only priors/thresholds/temperatures | unknown-only no-negative-gradient test |
| `covariate_homogenizer.py` | common covariate representation | field ablation test |
| `multiscale_pattern_router.py` | 1×/2×/4× mixer and four named patterns | stable/forming/releasing/oscillating tests |
| `mesoscopic_lane_flow.py` | distinct left/center/right flow descriptors | lane-difference tests |
| `traffic_state_reasoner.py` | exact 13 factors, low-rank composition, lineage | evidence reconstruction test |
| `response_lag.py` | causal lag 0–3 | delayed-response recovery |
| `global_decision_stream.py` | independent 32-step global control | target-independence test |
| `decision_ledger.py` | exact per-factor speed/course contributions | exact-sum identity |
| `contribution_alignment.py` | real contribution distribution and JS loss | perturb contribution changes loss |
| `text_decoder.py` | independent BERT, separate segment losses, no GT inference | gradient-direction test |
| `interventions.py` | earliest-layer re-forward interventions | real output deltas |
| `model.py` | orchestrates typed components, no monolithic hidden shortcuts | import graph and full smoke |

## Engines

| File | Required behavior |
|---|---|
| `acpr_dynflow_data.py` | direct-image loader, raw text metadata, aligned caption/control IDs, no val formal mode |
| `eval_adapt_reference.py` | separate comparison-only ADAPT evaluation |
| `train_acpr_dynflow.py` | one fixed 20-epoch run, optimizer/scheduler/resume, per-epoch test |
| `eval_acpr_dynflow.py` | exact ADAPT text/control metrics plus middle outputs |
| `run_acpr_dynflow_preflight.py` | orchestrates all blocking gates |
| `audit_acpr_dynflow.py` | writes all reports and review pass only on full success |
| `probe_acpr_dynflow_memory.py` | direct-image 30-step memory candidates |
| `export_acpr_dynflow_visuals.py` | real Dynamic Traffic Decision Ledger |
| `build_acpr_dynflow_atlas.py` | standalone HTML/JSON atlas |
| `supervise_acpr_dynflow_foreground.py` | attached complete-suite supervisor |

## Existing shared files

| File | Allowed modification |
|---|---|
| `src/modeling/load_swin.py` | backward-compatible stage-return/load helper only |
| `src/modeling/video_swin/swin_transformer.py` | backward-compatible native stage output only |
| `src/layers/bert/modeling_bert.py` | only if required for independent decoder hidden/logprob interface; preserve old APIs |
| `src/datasets/vision_language_tsv.py` | preserve raw metadata |
| `src/datasets/vl_dataloader.py` | collate raw strings and sample IDs |
| `src/datasets/caption_tensorizer.py` | parity fix only; no ACPR-specific logic |
| `.gitignore` | ignore run artifacts, not source |

## Forbidden implementation substitutions

- anonymous predicates;
- one global state vector followed by 13 decorative classifiers;
- repeated left/center/right values;
- post-hoc contribution estimates;
- ADAPT output residuals;
- duplicated action/explanation loss;
- fixed unknown-negative BCE;
- fake SCST/reward;
- placeholder canvas;
- correlation-only traffic-flow proof;
- YAML-only features;
- review based only on static scans.
