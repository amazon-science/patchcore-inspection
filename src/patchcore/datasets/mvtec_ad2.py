"""Dataset adapter for the MVTec AD 2 public split layout."""

import os
from pathlib import Path

import PIL
import torch
from torchvision import transforms

from patchcore.datasets.mvtec import DatasetSplit, IMAGENET_MEAN, IMAGENET_STD

_CLASSNAMES = ["sheet_metal"]
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class MVTecAD2Dataset(torch.utils.data.Dataset):
    """PyTorch Dataset for MVTec AD 2 public train/test data."""

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        image_transform=None,
        **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.split_name = split.value if isinstance(split, DatasetSplit) else split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.image_transform = image_transform
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
        ]
        if image_transform is not None:
            self.transform_img.append(image_transform)
        self.transform_img.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.transform_img = transforms.Compose(self.transform_img)
        self.transform_mask = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
            ]
        )
        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)
        if self.split_name == DatasetSplit.TEST.value and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = (self.transform_mask(mask) > 0).float()
        else:
            mask = torch.zeros([1, *image.size()[1:]])
        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(Path(image_path).parts[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        data_to_iterate = []
        for classname in self.classnames_to_use:
            class_root = os.path.join(self.source, classname)
            imgpaths_per_class[classname] = {}
            if self.split_name == DatasetSplit.TRAIN.value:
                imgpaths_per_class[classname]["good"] = self._image_files(
                    os.path.join(class_root, "train", "good")
                )
            elif self.split_name == DatasetSplit.VAL.value:
                imgpaths_per_class[classname]["good"] = self._image_files(
                    os.path.join(class_root, "validation", "good")
                )
            elif self.split_name == DatasetSplit.TEST.value:
                test_root = os.path.join(class_root, "test_public")
                imgpaths_per_class[classname]["good"] = self._image_files(
                    os.path.join(test_root, "good")
                )
                imgpaths_per_class[classname]["bad"] = self._image_files(
                    os.path.join(test_root, "bad")
                )
            elif self.split_name == "test_private":
                imgpaths_per_class[classname]["unknown"] = self._image_files(
                    os.path.join(class_root, "test_private")
                )
            elif self.split_name == "test_private_mixed":
                imgpaths_per_class[classname]["unknown"] = self._image_files(
                    os.path.join(class_root, "test_private_mixed")
                )
            else:
                raise ValueError("Unsupported split: {}".format(self.split_name))

            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for image_path in imgpaths_per_class[classname][anomaly]:
                    mask_path = None
                    if self.split_name == DatasetSplit.TEST.value and anomaly != "good":
                        mask_path = self._mask_path(class_root, image_path)
                    data_to_iterate.append([classname, anomaly, image_path, mask_path])
        return imgpaths_per_class, data_to_iterate

    @staticmethod
    def _image_files(directory):
        if not os.path.isdir(directory):
            raise FileNotFoundError("Dataset directory not found: {}".format(directory))
        return sorted(
            os.path.join(directory, filename)
            for filename in os.listdir(directory)
            if filename.lower().endswith(_IMAGE_EXTENSIONS)
        )

    @staticmethod
    def _mask_path(class_root, image_path):
        image = Path(image_path)
        mask_name = "{}_mask{}".format(image.stem, image.suffix)
        mask_path = os.path.join(
            class_root, "test_public", "ground_truth", "bad", mask_name
        )
        if not os.path.isfile(mask_path):
            raise FileNotFoundError("Ground-truth mask not found: {}".format(mask_path))
        return mask_path
