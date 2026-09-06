from unittest import mock

import pytest
import torch

from patchcore.device import DeviceManager
from patchcore.utils import set_torch_device


def test_explicit_cpu_does_not_require_cuda():
    with mock.patch("torch.cuda.is_available", return_value=True):
        assert DeviceManager().resolve("cpu") == torch.device("cpu")


def test_auto_falls_back_to_cpu_when_cuda_is_unavailable():
    with mock.patch("torch.cuda.is_available", return_value=False):
        assert DeviceManager().resolve("auto", [0]) == torch.device("cpu")


def test_explicit_cuda_fails_when_cuda_is_unavailable():
    with mock.patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA was requested"):
            DeviceManager().resolve("cuda", [0])


def test_auto_uses_requested_gpu_when_cuda_is_available():
    with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
        "torch.cuda.device_count", return_value=2
    ):
        assert DeviceManager().resolve("auto", [1]) == torch.device("cuda:1")


def test_invalid_device_name_is_rejected():
    with pytest.raises(ValueError, match="Unsupported device"):
        DeviceManager().resolve("tpu")


def test_legacy_empty_gpu_list_still_selects_cpu():
    assert set_torch_device([]) == torch.device("cpu")


def test_legacy_gpu_selection_falls_back_to_cpu():
    with mock.patch("torch.cuda.is_available", return_value=False):
        assert set_torch_device([0]) == torch.device("cpu")
