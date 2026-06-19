# ACPR-FlowCal++ V1 Implementation Plan

This runbook is the repository copy of the user-supplied ACPR-FlowCal++ V1 contract.

Formal training is blocked until:

- `python -m compileall -q fate_x src` passes.
- `python -m pytest tests/test_acpr_flow_*.py -q` passes.
- `python -m fate_x.engine.audit_acpr_flowcal_pp --config configs/acpr_flowcal_pp_v1_bddx_32f_224.yaml --output_dir .background_runs/acpr_flowcal_pp_v1_preflight --device cuda --write_review_pass` writes `REVIEW_PASS_ACPR_FLOWCAL_PP_V1.txt` on a clean pushed `flowtrace_pmt_v1` HEAD.

The formal path is direct-image/no-cache and must not use the legacy FlowTrace PMT path.
