from fate_x.engine.acpr_flowcal_v2_data import build_v2_dataloader, resolve_adapt_text_contract


def test_adapt_text_contract_uses_official_bddx_defaults():
    c = resolve_adapt_text_contract()
    assert c["mask_prob"] == 0.5
    assert c["max_masked_tokens"] == 45
    assert c["max_seq_length"] == 30


def test_validation_loader_is_forbidden_in_formal_mode():
    try:
        build_v2_dataloader("validation", formal=True)
    except ValueError:
        return
    raise AssertionError("validation loader must be rejected")

def test_formal_loader_uses_real_bddx_32frame_tsv_not_synthetic():
    loader = build_v2_dataloader(
        "train",
        batch_size=1,
        num_workers=0,
        formal=True,
        synthetic=False,
        length=2,
        data_dir="datasets_part",
        yaml_file="BDDX/training_32frames.yaml",
    )
    batch = next(iter(loader))
    assert batch.frames.shape == (1, 32, 3, 224, 224)
    assert batch.car_info.shape == (1, 2, 32)
    assert batch.sample_ids[0].startswith("training_")
    assert not batch.sample_ids[0].startswith("synthetic_")
    assert batch.raw_actions[0]
    assert batch.raw_justifications[0]
