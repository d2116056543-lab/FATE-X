# FATE-X ADAPT Compatibility

Baseline behavior must remain available with all FATE flags disabled:

- `--fate_x_enabled false`
- `--video_token_reducer none`
- `--temporal_evidence_memory none`
- `--phrase_faithfulness_enabled false`

This worktree currently adds standalone FATE-X modules and wrapper scripts. Direct integration into
`src/tasks/run_adapt.py` is intentionally conservative because ADAPT's exact Linux/Apex/DeepSpeed path
must remain reproducible. The token reducer/event memory modules are ready for insertion after ADAPT's
video token projection once the next training stage is approved.
