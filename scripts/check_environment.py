"""Report the runtime environment used by PatchCore experiments."""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchvision


class EnvironmentInspector:
    """Collect environment metadata without assuming CUDA or FAISS exists."""

    def __init__(self, repository_root: Path) -> None:
        self._repository_root = repository_root

    def inspect(self) -> Dict[str, Any]:
        cuda_available = torch.cuda.is_available()
        return {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "os": platform.platform(),
            "architecture": platform.machine(),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "cuda_compiled_version": torch.version.cuda,
            "cuda_available": cuda_available,
            "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
            "numpy_version": np.__version__,
            "faiss_version": self._faiss_version(),
            "git_commit": self._git_commit(),
        }

    @staticmethod
    def _faiss_version() -> Optional[str]:
        try:
            import faiss
        except ImportError:
            return None
        return getattr(faiss, "__version__", "unknown")

    def _git_commit(self) -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=self._repository_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. Parent directories are created.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    report = EnvironmentInspector(repository_root).inspect()
    serialized_report = json.dumps(report, indent=2, sort_keys=True)
    print(serialized_report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized_report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
