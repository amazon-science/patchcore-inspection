import numpy as np
import pytest

from patchcore.metrics import compute_aupro


def test_aupro_is_one_for_perfectly_separated_regions():
    masks = np.zeros((2, 4, 4), dtype=np.uint8)
    masks[0, 0:2, 0:2] = 1
    masks[1, 2:4, 2:4] = 1
    predictions = np.where(masks == 1, 1.0, 0.0)

    result = compute_aupro(predictions, masks, fpr_limit=0.3)

    assert result["aupro"] == pytest.approx(1.0)
    assert result["number_of_regions"] == 2


def test_aupro_is_zero_when_background_ranks_before_anomalies_at_limit():
    masks = np.zeros((1, 10, 10), dtype=np.uint8)
    masks[0, 0:2, 0:2] = 1
    predictions = np.where(masks == 1, 0.0, 1.0)

    result = compute_aupro(predictions, masks, fpr_limit=0.3)

    assert result["aupro"] == pytest.approx(0.0)


def test_aupro_rejects_data_without_anomalous_regions():
    with pytest.raises(ValueError, match="anomalous ground-truth region"):
        compute_aupro(np.zeros((1, 4, 4)), np.zeros((1, 4, 4)))
