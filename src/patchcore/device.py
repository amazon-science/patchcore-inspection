"""Central runtime-device selection for PatchCore."""

from typing import Optional, Sequence

import torch


class DeviceManager:
    """Resolve CPU/CUDA devices without leaking availability checks to callers."""

    _SUPPORTED_DEVICES = {"auto", "cpu", "cuda"}

    def resolve(
        self,
        requested_device: Optional[str] = "auto",
        gpu_ids: Optional[Sequence[int]] = None,
    ) -> torch.device:
        requested = (requested_device or "auto").lower()
        if requested not in self._SUPPORTED_DEVICES:
            supported = ", ".join(sorted(self._SUPPORTED_DEVICES))
            raise ValueError(
                "Unsupported device '{}'. Expected one of: {}.".format(
                    requested_device, supported
                )
            )

        if requested == "cpu":
            return torch.device("cpu")

        cuda_available = torch.cuda.is_available()
        if requested == "cuda" and not cuda_available:
            raise RuntimeError("CUDA was requested but is not available.")

        if cuda_available:
            gpu_id = gpu_ids[0] if gpu_ids else 0
            if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
                raise ValueError(
                    "CUDA device {} is outside the available range [0, {}).".format(
                        gpu_id, torch.cuda.device_count()
                    )
                )
            return torch.device("cuda:{}".format(gpu_id))

        return torch.device("cpu")
