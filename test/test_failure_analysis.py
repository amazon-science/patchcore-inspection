import numpy as np
import pytest

from scripts.analyze_failures import PairedFailureAnalyzer


def test_inside_outside_delta_uses_ground_truth_mask():
    delta = np.array([[[2.0, 1.0], [1.0, 1.0]]])
    mask = np.array([[[[1, 0], [0, 0]]]])

    inside, outside = PairedFailureAnalyzer._inside_outside_delta(delta, mask)

    assert inside[0] == pytest.approx(2.0)
    assert outside[0] == pytest.approx(1.0)


def test_pairing_rejects_different_image_order():
    baseline = {
        "image_paths": np.array(["a.png", "b.png"]),
        "labels": np.array([0, 1]),
        "masks": np.zeros((2, 2, 2)),
    }
    candidate = {
        "image_paths": np.array(["b.png", "a.png"]),
        "labels": baseline["labels"],
        "masks": baseline["masks"],
    }

    with pytest.raises(ValueError, match="ordered image paths"):
        PairedFailureAnalyzer._validate_pairing(baseline, candidate)
