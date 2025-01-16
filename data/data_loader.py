import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import torch
import torchvision.transforms.v2 as v2
from torchvision.datasets import CocoDetection, VisionDataset, VOCDetection
from torchvision.transforms.v2 import Transform

from data.dataset import Dataset
from training.encoder import CenternetEncoder
from utils.config import IMG_HEIGHT, IMG_WIDTH
from utils.io_utils import download_file, unzip_archive


def convert_voc_annotations(target):
    """Converts VOC XML annotations to the required format"""
    objects = target["annotation"]["object"]
    if not isinstance(objects, list):
        objects = [objects]

    boxes = []
    labels = []

    for obj in objects:
        try:
            bbox = obj["bndbox"]
            x1 = float(bbox["xmin"])
            y1 = float(bbox["ymin"])
            x2 = float(bbox["xmax"])
            y2 = float(bbox["ymax"])

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(1)  # За замовчуванням всі об'єкти одного класу
        except (KeyError, ValueError) as e:
            print(f"Error processing VOC annotation: {e}")
            continue

    if not boxes:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros(0, dtype=torch.int64),
        }

    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def convert_coco_annotations(target):
    """Converts COCO annotations to VOC format"""
    if not isinstance(target, list):
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros(0, dtype=torch.int64),
        }

    boxes = []
    labels = []

    for annotation in target:
        if not isinstance(annotation, dict):
            continue

        bbox = annotation.get("bbox")
        category_id = annotation.get("category_id")

        if bbox is None or category_id is None:
            continue

        try:
            # COCO format: [x, y, width, height] to [x1, y1, x2, y2]
            x1 = float(bbox[0])
            y1 = float(bbox[1])
            w = float(bbox[2])
            h = float(bbox[3])
            x2 = x1 + w
            y2 = y1 + h

            if w <= 0 or h <= 0:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(category_id)
        except (ValueError, TypeError, IndexError) as e:
            print(f"Error processing bbox {bbox}: {e}")
            continue

    if not boxes:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros(0, dtype=torch.int64),
        }

    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


class DataLoader(ABC):
    def __init__(self, *, dataset_path: str, image_set: str = None):
        self.image_set = image_set or "train"
        self.dataset_path = dataset_path

    @abstractmethod
    def load(
        self,
        transforms: Optional[Transform] = None,
        encoder: Optional[Callable] = None,
    ) -> Dataset:
        """
        Loads data and returns pytorch VisionDataset.
        The dataset is automatically downloaded
        if `dataset_path` does not exist
        """

    def _post_load_hook(
        self,
        dataset: VisionDataset,
        transforms: Optional[Transform] = None,
        encoder: Optional[Callable] = None,
    ) -> Dataset:
        transforms = transforms or v2.Compose(
            [
                v2.Resize(size=(IMG_WIDTH, IMG_HEIGHT), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        encoder = encoder or CenternetEncoder(IMG_HEIGHT, IMG_WIDTH)
        return Dataset(dataset, transforms, encoder)


class PascalVOCDataLoader(DataLoader, VOCDetection):
    def __init__(self, *, dataset_path: str, image_set: str = None):
        DataLoader.__init__(self, dataset_path=dataset_path, image_set=image_set)
        self.is_download = not os.path.exists(self.dataset_path)
        VOCDetection.__init__(
            self,
            root=self.dataset_path,
            year="2007",
            image_set=self.image_set,
            download=self.is_download,
        )

    def __getitem__(self, index):
        img, target = VOCDetection.__getitem__(self, index)
        return img, convert_voc_annotations(target)

    def load(
        self,
        transforms: Optional[Transform] = None,
        encoder: Optional[Callable] = None,
    ):
        return post_load_hook(self, transforms, encoder)


class MSCocoDataLoader(DataLoader, CocoDetection):
    DATASET_URLS = {
        "train": {
            "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            "images": "http://images.cocodataset.org/zips/train2017.zip",
            "ann_file": "instances_train2017.json",
        },
        "val": {
            "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            "images": "http://images.cocodataset.org/zips/val2017.zip",
            "ann_file": "instances_val2017.json",
        },
        "test": {
            "annotations": "http://images.cocodataset.org/annotations/image_info_test2017.zip",
            "images": "http://images.cocodataset.org/zips/test2017.zip",
            "ann_file": "image_info_test2017.json",
        },
    }

    def __init__(self, *, dataset_path: str, image_set: str = None):
        DataLoader.__init__(self, dataset_path=dataset_path, image_set=image_set)
        self.ann_folder = Path(self.dataset_path) / "annotations"
        self.images_folder = Path(self.dataset_path) / f"{self.image_set}2017"

        if not os.path.exists(self.dataset_path):
            os.makedirs(self.ann_folder, exist_ok=True)
            os.makedirs(self.images_folder, exist_ok=True)

            file_urls = [
                self.DATASET_URLS[self.image_set]["annotations"],
                self.DATASET_URLS[self.image_set]["images"],
            ]
            for url in file_urls:
                filepath = Path(self.dataset_path) / url.split("/")[-1]
                download_file(url, filepath)
                unzip_archive(filepath, self.dataset_path)

            print(f"\t\t{self.image_set} dataset is downloaded")

        CocoDetection.__init__(
            self,
            root=str(self.images_folder),
            annFile=str(
                self.ann_folder / self.DATASET_URLS[self.image_set]["ann_file"]
            ),
        )

    def __getitem__(self, index):
        img, target = CocoDetection.__getitem__(self, index)
        return img, convert_coco_annotations(target)

    def load(
        self,
        transforms: Optional[Transform] = None,
        encoder: Optional[Callable] = None,
    ):
        return post_load_hook(self, transforms, encoder)
