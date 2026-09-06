from pathlib import Path

import torch

from patchcore.config import ConfigLoader
from patchcore.experiment import ExperimentRunner, PatchCoreFactory


def test_factory_builds_frozen_cpu_model_from_shared_config():
    root = Path(__file__).resolve().parents[1]
    config = ConfigLoader().load(root / "configs/patchcore_cpu_smoke.yaml")
    model = PatchCoreFactory().create(config, torch.device("cpu"))

    assert model.device.type == "cpu"
    assert model.layers_to_extract_from == ["layer2", "layer3"]
    assert not model.backbone.training
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())


def test_run_directory_contains_experiment_and_class(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = ConfigLoader().load(root / "configs/patchcore_cpu_smoke.yaml")

    run_directory = ExperimentRunner._create_run_directory(tmp_path, config)

    assert run_directory.parent == tmp_path
    assert "patchcore_cpu_smoke_bottle_" in run_directory.name
