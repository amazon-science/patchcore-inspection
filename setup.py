from pathlib import Path

from setuptools import find_packages
from setuptools import setup

package_path = Path(__file__).parent

version_path = Path(__file__).parent.joinpath(f"src/patchcore/VERSION")
version = version_path.read_text().strip()
install_requires = (
    Path(__file__).parent.joinpath("requirements.txt").read_text().splitlines()
)
data_files = [
    str(version_path.relative_to(package_path)),
]

setup(
    name="patchcore",
    version=version,
    install_requires=install_requires,
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    data_files=data_files,
)
