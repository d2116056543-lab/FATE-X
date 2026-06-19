import pytest
import torch

from fate_x.acpr_flow.sequence_calalign import SequenceCalAlign


def test_sequence_calalign_allows_zero_alpha_and_rejects_test_fit():
    fitter = SequenceCalAlign(["train_calib_0", "train_calib_1"])
    base = torch.zeros(2, 3)
    enh = torch.randn(2, 3)
    target = torch.zeros(2, dtype=torch.long)
    scales = fitter.fit(["train_calib_0", "train_calib_1"], base, enh, target, alpha_grid=[0.0], temperature_grid=[1.0])
    assert scales.alpha_action == 0.0
    assert torch.allclose(fitter.apply_action(base, enh), base)
    with pytest.raises(ValueError):
        fitter.fit(["test_0"], base[:1], enh[:1], target[:1])
