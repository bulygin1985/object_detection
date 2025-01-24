"""
Convert VOC format predictions to COCO.
Example to run:
python predictions_voc2coco.py
"""

import json
import os
from collections import namedtuple

import torch
import torchvision
from load_model import load_model
from predictions import get_predictions
from torch.utils.data import Subset
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

from data.dataset import Dataset
from models.centernet import ModelBuilder

input_height = input_width = 256
Img_info = namedtuple("Img_info", ["id", "file_name", "height", "width"])


def prepare_dataset():
    # Load VOC dataset
    dataset = torchvision.datasets.VOCDetection(
        root="../VOC", year="2007", image_set="trainval", download=False
    )

    dataset_val = torchvision.datasets.wrap_dataset_for_transforms_v2(dataset)

    # Define a dataset that is a subset of the initial dataset
    indices = range(10)
    dataset_val = Subset(dataset_val, indices)

    imgs_info = []
    for i in indices:
        ann = dataset[i][1]["annotation"]
        filename = ann["filename"]
        id_str, _ = os.path.splitext(filename)
        imgs_info.append(
            Img_info(
                id=int(id_str),
                file_name=filename,
                height=ann["size"]["height"],
                width=ann["size"]["width"],
            )
        )
    # the same can be achieved much easier if dataset is a CocoDetection dataset:
    # img_info = dataset.coco.dataset['images']

    return {"images_info": imgs_info, "annotations": dataset_val}


def convert_predictions_to_coco_format(img_filenames, preds, output_path: str = None):
    # [{
    #     "image_id": int,
    #     "category_id": int,
    #     "bbox": [x, y, width, height],
    #     "score": float,
    # }]

    results = []

    pred_shape = preds[0].squeeze().shape
    num_categories = pred_shape[0] - 4

    total = len(img_filenames)

    with tqdm(total=total) as pbar:
        for filename, pred in zip(img_filenames, preds):
            squeezed_pred = pred.squeeze()
            for category in range(num_categories):
                for i in range(pred_shape[1]):
                    for j in range(pred_shape[2]):
                        img_id, _ = os.path.splitext(filename)
                        box = squeezed_pred[num_categories:, i, j].tolist()
                        results.append(
                            {
                                "image_id": int(img_id),
                                "category_id": category,
                                "bbox": [
                                    box[0],
                                    box[1],
                                    box[2] - box[0],
                                    box[3] - box[1],
                                ],
                                "score": squeezed_pred[category, i, j].item(),
                            }
                        )
            pbar.update(1)

    print("Store result in file: ", output_path, "\n")
    if output_path is not None:
        with open(output_path, "w") as f:
            json.dump(results, f)

    return results


def transform_dataset(dataset):
    """Transform the dataset for visualization"""
    transform = transforms.Compose(
        [
            transforms.Resize(size=(input_width, input_height)),
        ]
    )

    return Dataset(dataset=dataset, transformation=transform)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(device, ModelBuilder)

    dataset = prepare_dataset()
    dataset_transformed = transform_dataset(dataset["annotations"])
    predictions = get_predictions(device, model, dataset_transformed)

    img_filenames = [elem.file_name for elem in dataset["images_info"]]

    _ = convert_predictions_to_coco_format(
        img_filenames, predictions, "../VOC_COCO/pascal_trainval2007_predictions.json"
    )
