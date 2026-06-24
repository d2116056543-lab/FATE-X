
def test_best_selectors_text_control_and_test_floor():
    from fate_x.engine.eval_acpr_dynflow_swin import select_best_records

    records = [
        {"epoch": 0, "CIDEr_description": 1.0, "CIDEr_explanation": 0.5, "speed_RMSE": 3.0, "course_RMSE": 6.0},
        {"epoch": 1, "CIDEr_description": 1.2, "CIDEr_explanation": 0.7, "speed_RMSE": 2.0, "course_RMSE": 5.0},
    ]
    best = select_best_records(records, adapt_reference={"CIDEr_sum": 2.0, "speed_RMSE": 2.5, "course_RMSE": 5.5})
    assert best["text"]["epoch"] == 1
    assert best["control"]["epoch"] == 1
    assert best["test"]["epoch"] == 1
