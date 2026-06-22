from pathlib import Path
p=Path(r'E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree\fate_x\engine\train_acpr_flowcal_v2.py')
t=p.read_text(encoding='utf-8')
if 'def _append_jsonl_with_retry' not in t:
    t=t.replace('def _save(path: Path, model, optimizer, epoch, global_step, metrics):\n', '''def _append_jsonl_with_retry(path: Path, record: dict, retries: int = 3) -> None:\n    payload = json.dumps(record) + "\\n"\n    for attempt in range(retries):\n        try:\n            path.parent.mkdir(parents=True, exist_ok=True)\n            with path.open("a", encoding="utf-8") as handle:\n                handle.write(payload)\n            return\n        except OSError as exc:\n            if attempt + 1 >= retries:\n                print("ACPR_FLOWCAL_V2_LOG_WRITE_SKIPPED " + json.dumps({"path": str(path), "error": str(exc)}), flush=True)\n                return\n            time.sleep(0.5 * (attempt + 1))\n\n\ndef _save(path: Path, model, optimizer, epoch, global_step, metrics):\n''')
    old='''    path.parent.mkdir(parents=True, exist_ok=True)\n    torch.save(\n        {\n            "model": model.state_dict(),\n            "optimizer": optimizer.state_dict(),\n            "epoch": epoch,\n            "global_step": global_step,\n            "metrics": metrics,\n        },\n        path,\n    )\n'''
    new='''    payload = {\n        "model": model.state_dict(),\n        "optimizer": optimizer.state_dict(),\n        "epoch": epoch,\n        "global_step": global_step,\n        "metrics": metrics,\n    }\n    for attempt in range(3):\n        try:\n            path.parent.mkdir(parents=True, exist_ok=True)\n            torch.save(payload, path)\n            return\n        except OSError as exc:\n            if attempt == 2:\n                raise\n            print("ACPR_FLOWCAL_V2_CKPT_RETRY " + json.dumps({"path": str(path), "error": str(exc), "attempt": attempt + 1}), flush=True)\n            time.sleep(1.0 * (attempt + 1))\n'''
    t=t.replace(old,new)
    old2='''            with loss_log.open("a", encoding="utf-8") as handle:\n                handle.write(json.dumps(rec) + "\\n")\n            if batch_idx % max(1, args.log_every) == 0:\n                print("ACPR_FLOWCAL_V2_BATCH " + json.dumps(rec), flush=True)\n'''
    new2='''            if batch_idx % max(1, args.log_every) == 0:\n                _append_jsonl_with_retry(loss_log, rec)\n                print("ACPR_FLOWCAL_V2_BATCH " + json.dumps(rec), flush=True)\n'''
    t=t.replace(old2,new2)
    old3='''        with metrics_log.open("a", encoding="utf-8") as handle:\n            handle.write(json.dumps(metrics) + "\\n")\n'''
    new3='''        _append_jsonl_with_retry(metrics_log, metrics)\n'''
    t=t.replace(old3,new3)
p.write_text(t, encoding='utf-8')
print('patched train_acpr_flowcal_v2 safe logging')
