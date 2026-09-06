"""Dataset factory shared by experiment and smoke scripts."""

from patchcore.datasets.mvtec import DatasetSplit, MVTecDataset
from patchcore.datasets.mvtec_ad2 import MVTecAD2Dataset

_DATASETS = {
    "mvtec": MVTecDataset,
    "mvtec_ad_2": MVTecAD2Dataset,
}


def create_dataset(dataset_name, split=DatasetSplit.TRAIN, **kwargs):
    try:
        dataset_class = _DATASETS[dataset_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_DATASETS))
        raise ValueError(
            "Unsupported dataset.name={}. Supported datasets: {}".format(
                dataset_name, supported
            )
        ) from exc
    return dataset_class(split=split, **kwargs)
