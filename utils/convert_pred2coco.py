"""
Convert predictions to COCO format.
Example to run:
python convert_pred2coco.py
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

IMG_HEIGHT = IMG_WIDTH = 256

Img_info = namedtuple("Img_info", ["id", "filename", "height", "width"])


def prepare_dataset():
    # Load VOC dataset
    dataset = torchvision.datasets.VOCDetection(
        root="../VOC", year="2007", image_set="train", download=False
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
                filename=filename,
                height=int(ann["size"]["height"]),
                width=int(ann["size"]["width"]),
            )
        )
    # the same can be achieved much easier if dataset is a CocoDetection dataset:
    # img_info = dataset.coco.dataset['images']

    return {"images_info": imgs_info, "annotations": dataset_val}


def convert_predictions_to_coco_format(
    imgs_info: list[Img_info],
    preds,
    output_stride_h: int = 4,
    output_stride_w: int = 4,
    output_path: str = None,
) -> list[dict[str, object]]:
    # [{
    #     "image_id": int,
    #     "category_id": int,
    #     "bbox": [x, y, width, height],
    #     "score": float,
    # }]

    results = []

    pred_shape = preds[0].shape
    num_categories = pred_shape[0] - 4

    total = len(imgs_info)

    with tqdm(total=total) as pbar:
        for img_info, pred in zip(imgs_info, preds):
            width_scale_factor = float(img_info.width / IMG_WIDTH)
            height_scale_factor = float(img_info.height / IMG_HEIGHT)

            # get image id from the filename
            rev_filename = "".join(reversed(img_info.filename))
            rev_id_part_filename, _, _ = rev_filename.partition("_")
            id_part_filename = "".join(reversed(rev_id_part_filename))
            img_id_str, _ = os.path.splitext(id_part_filename)
            img_id = int(img_id_str)

            for category in range(num_categories):
                for i in range(pred_shape[1]):
                    for j in range(pred_shape[2]):
                        box = pred[num_categories:, i, j].tolist()
                        results.append(
                            {
                                "image_id": img_id,
                                "category_id": category + 1,
                                "bbox": [
                                    (j * output_stride_h - box[0]) * width_scale_factor,
                                    (i * output_stride_w - box[1])
                                    * height_scale_factor,
                                    (box[2] + box[0]) * width_scale_factor,
                                    (box[3] + box[1]) * height_scale_factor,
                                ],
                                "score": pred[category, i, j].item(),
                            }
                        )
            pbar.update(1)

    if output_path is not None:
        print(f"Storing result in file: {output_path}...")
        with open(output_path, "w") as f:
            json.dump(results, f)

    return results


def transform_dataset(dataset):
    """Transform the dataset for visualization"""
    transform = transforms.Compose(
        [
            transforms.Resize(size=(IMG_WIDTH, IMG_HEIGHT)),
        ]
    )

    return Dataset(dataset=dataset, transformation=transform)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(device, ModelBuilder, alpha=0.25)

    dataset = prepare_dataset()
    dataset_transformed = transform_dataset(dataset["annotations"])
    predictions = get_predictions(device, model, dataset_transformed)
    predictions = [pred.squeeze() for pred in predictions]

    num_categories = predictions[0].shape[0] - 4

    img_filenames = [elem.filename for elem in dataset["images_info"]]

    _ = convert_predictions_to_coco_format(
        dataset["images_info"],
        predictions,
        output_path="../VOC_COCO/pascal_train2007_predictions.json",
    )
