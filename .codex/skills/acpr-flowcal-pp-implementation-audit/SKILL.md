---
name: acpr-flowcal-pp-implementation-audit
description: Blocking audit for ACPR-FlowCal++ V1 formal training authorization.
---

# ACPR-FlowCal++ V1 Audit

Run:

```bash
python -m fate_x.engine.audit_acpr_flowcal_pp \
  --config configs/acpr_flowcal_pp_v1_bddx_32f_224.yaml \
  --output_dir .background_runs/acpr_flowcal_pp_v1_preflight \
  --device cuda \
  --write_review_pass
```

Do not authorize formal training unless the worktree is clean, the pushed GitHub branch matches local HEAD, the direct-image/no-cache smoke passes, and `REVIEW_PASS_ACPR_FLOWCAL_PP_V1.txt` is produced for that exact commit.
