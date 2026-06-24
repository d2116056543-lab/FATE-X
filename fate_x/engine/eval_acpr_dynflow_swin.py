from __future__ import annotations


def _cider_sum(record: dict) -> float:
    return float(record.get("CIDEr_description", 0.0)) + float(record.get("CIDEr_explanation", 0.0))


def _control_score(record: dict, adapt_reference: dict) -> float:
    speed_ref = max(float(adapt_reference.get("speed_RMSE", 1.0)), 1e-6)
    course_ref = max(float(adapt_reference.get("course_RMSE", 1.0)), 1e-6)
    return 0.5 * float(record.get("speed_RMSE", 1e9)) / speed_ref + 0.5 * float(record.get("course_RMSE", 1e9)) / course_ref


def select_best_records(records: list[dict], adapt_reference: dict) -> dict[str, dict]:
    if not records:
        raise ValueError("records must not be empty")
    best_text = max(records, key=_cider_sum)
    best_control = min(records, key=lambda r: _control_score(r, adapt_reference))
    floor = 0.85 * float(adapt_reference.get("CIDEr_sum", 0.0))
    eligible = [r for r in records if _cider_sum(r) >= floor]
    pool = eligible or records
    best_test = min(
        pool,
        key=lambda r: (
            _control_score(r, adapt_reference),
            -float(r.get("CIDEr_explanation", 0.0)),
            -float(r.get("CIDEr_description", 0.0)),
            float(r.get("speed_RMSE", 1e9)),
            float(r.get("course_RMSE", 1e9)),
        ),
    )
    return {
        "text": best_text,
        "control": best_control,
        "joint": best_test,
        "test": {**best_test, "text_floor_not_met": not bool(eligible)},
    }


def main() -> None:
    raise SystemExit("eval_acpr_dynflow_swin is implemented as a library entrypoint; full evaluator requires dataset assets")


if __name__ == "__main__":
    main()
