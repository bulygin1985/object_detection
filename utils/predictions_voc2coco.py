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
                file_name=filename,
                height=int(ann["size"]["height"]),
                width=int(ann["size"]["width"]),
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

    pred_shape = preds[0].shape
    num_categories = pred_shape[0] - 4

    total = len(img_filenames)

    with tqdm(total=total) as pbar:
        for filename, pred in zip(img_filenames, preds):
            for category in range(1, num_categories):
                for i in range(pred_shape[1]):
                    for j in range(pred_shape[2]):
                        img_id, _ = os.path.splitext(filename)
                        box = pred[num_categories:, i, j].tolist()
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
                                "score": pred[category, i, j].item(),
                            }
                        )
            pbar.update(1)

    print("Storing result in file: ", output_path, "...")
    if output_path is not None:
        with open(output_path, "w") as f:
            json.dump(results, f)

    return results


def get_inverse_resize_transformations(
    num_categories: int,
    ground_truth_images_info: list[Img_info],
    transformed_width: int,
    transformed_height: int,
):
    def get_bbox_resizer(num_cat, original_width, original_height):
        def bbox_resizer(pred):
            pred[num_cat, :, :] = pred[num_cat, :, :] * original_width / transformed_width
            pred[num_cat + 1, :, :] = pred[num_cat + 1, :, :] * original_height / transformed_height
            pred[num_cat + 2, :, :] = pred[num_cat + 2, :, :] * original_width / transformed_width
            pred[num_cat + 3, :, :] = pred[num_cat + 3, :, :] * original_height / transformed_height
            return pred

        return bbox_resizer

    transformations = []
    for img_info in ground_truth_images_info:
        transformations.append(get_bbox_resizer(
            num_categories,
            img_info.width,
            img_info.height
        ))

    return transformations


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
    predictions = [pred.squeeze() for pred in predictions]

    num_categories = predictions[0].shape[0] - 4

    inverse_transfomations = get_inverse_resize_transformations(
        num_categories,
        dataset["images_info"],
        transformed_width=input_width,
        transformed_height=input_height,
    )

    resized_predictions = [
        transform(pred)
        for transform, pred in zip(inverse_transfomations, predictions)
    ]

    img_filenames = [elem.file_name for elem in dataset["images_info"]]

    _ = convert_predictions_to_coco_format(
        img_filenames, predictions, "../VOC_COCO/pascal_train2007_predictions.json"
    )
