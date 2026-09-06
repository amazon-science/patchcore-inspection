from pathlib import Path

from patchcore.config import ConfigLoader


def test_baseline_config_loads_with_expected_frozen_values():
    repository_root = Path(__file__).resolve().parents[1]
    config = ConfigLoader().load(repository_root / "configs/patchcore_baseline.yaml")

    assert config.experiment.seed == 42
    assert config.device.requested == "auto"
    assert config.dataset.name == "mvtec"
    assert config.dataset.class_name == "bottle"
    assert config.dataset.image_size == 224
    assert config.patchcore.backbone == "wideresnet50"
    assert config.patchcore.feature_layers == ("layer2", "layer3")
    assert config.patchcore.coreset_ratio == 0.1
    assert config.patchcore.nearest_neighbors == 1
    assert config.evaluation.au_pro is True
    assert config.augmentation.enabled is False
    assert config.augmentation.brightness == (1.0, 1.0)
    assert config.test_shift.enabled is False
    assert config.test_shift.kind == "none"


def test_illumination_config_changes_only_augmentation_settings():
    repository_root = Path(__file__).resolve().parents[1]
    baseline = ConfigLoader().load(repository_root / "configs/patchcore_baseline.yaml")
    augmented = ConfigLoader().load(
        repository_root / "configs/experiments/illumination_aug.yaml"
    )

    assert augmented.experiment.seed == baseline.experiment.seed
    assert augmented.dataset == baseline.dataset
    assert augmented.patchcore == baseline.patchcore
    assert augmented.evaluation == baseline.evaluation
    assert augmented.augmentation.enabled is True
    assert augmented.augmentation.brightness == (0.8, 1.2)
    assert augmented.augmentation.contrast == (0.8, 1.2)
    assert augmented.augmentation.gamma == (0.8, 1.2)


def test_controlled_brightness_configs_change_only_train_method_and_test_shift():
    repository_root = Path(__file__).resolve().parents[1]
    vanilla = ConfigLoader().load(
        repository_root / "configs/experiments/vanilla_brightness_08.yaml"
    )
    augmented = ConfigLoader().load(
        repository_root / "configs/experiments/augmented_brightness_08.yaml"
    )

    assert vanilla.dataset == augmented.dataset
    assert vanilla.patchcore == augmented.patchcore
    assert vanilla.evaluation == augmented.evaluation
    assert vanilla.test_shift == augmented.test_shift
    assert vanilla.test_shift.kind == "brightness"
    assert vanilla.test_shift.factor == 0.8
    assert vanilla.augmentation.enabled is False
    assert augmented.augmentation.enabled is True
