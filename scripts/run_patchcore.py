"""Run one reproducible, config-driven PatchCore experiment."""

import argparse
from pathlib import Path

from patchcore.experiment import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("experiments"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    run_directory = ExperimentRunner(repository_root).run(args.config, args.output_root)
    print("run_directory: {}".format(run_directory))
    print("PASS")


if __name__ == "__main__":
    main()
