from unittest import mock

import numpy as np
from PIL import Image

from patchcore.augmentation import FixedIlluminationTransform
from patchcore.augmentation import RandomIlluminationTransform
from patchcore.config import IlluminationAugmentationConfig


def test_identity_illumination_transform_preserves_pixels():
    image = Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8))
    transform = RandomIlluminationTransform(IlluminationAugmentationConfig())

    transformed = transform(image)

    np.testing.assert_array_equal(np.asarray(transformed), np.asarray(image))


def test_illumination_transform_applies_all_three_sampled_factors():
    image = Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8))
    config = IlluminationAugmentationConfig(
        enabled=True,
        brightness=(0.8, 1.2),
        contrast=(0.8, 1.2),
        gamma=(0.8, 1.2),
    )
    transform = RandomIlluminationTransform(config)

    with mock.patch("patchcore.augmentation.random.uniform") as sample:
        sample.side_effect = [0.8, 1.2, 0.9]
        transformed = transform(image)

    assert sample.call_args_list == [
        mock.call(0.8, 1.2),
        mock.call(0.8, 1.2),
        mock.call(0.8, 1.2),
    ]
    assert transformed.size == image.size
    assert transform.samples == [{"brightness": 0.8, "contrast": 1.2, "gamma": 0.9}]


def test_fixed_brightness_shift_is_deterministic_and_keeps_geometry():
    pixels = np.full((8, 8, 3), 100, dtype=np.uint8)
    image = Image.fromarray(pixels)
    transform = FixedIlluminationTransform("brightness", 0.8)

    first = transform(image)
    second = transform(image)

    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    assert first.size == image.size
    assert np.asarray(first).mean() == 80
