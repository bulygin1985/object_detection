"""
Convert predictions to COCO format.
Example to run:
python convert_pred2coco.py "../VOC/VOCdevkit/VOC2007/JPEGImages" \
    "../VOC_COCO/pascal_trainval2007.json" \
    --output_file="../VOC_COCO/pascal_train2007_predictions.json" \
    --imgs_ids="12, 17, 23, 26, 32, 33, 34, 35, 36, 42"
"""

import argparse
import json
from typing import Any

import torch
from load_model import load_model
from predictions import get_predictions
from torch.utils.data import Subset
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

from data.dataset import Dataset
from data.dataset_loaders import MSCOCODatasetLoader
from models.centernet import ModelBuilder

BBOX_PART_LEN = 4
IMG_HEIGHT = IMG_WIDTH = 256


def prepare_dataset(imgs_dir: str, ann_file: str, imgs_ids: list[int] = None):
    # Load COCO dataset
    ds_loader = MSCOCODatasetLoader(imgs_dir, ann_file)

    dataset = ds_loader.get_dataset()
    imgs_info = dataset.coco.dataset["images"]

    dataset = sorted(dataset, key=lambda x: int(x[1]["image_id"]))
    imgs_info = sorted(imgs_info, key=lambda x: int(x["id"]))

    subset = (
        {
            i: img_info
            for i, img_info in enumerate(imgs_info)
            if img_info["id"] in imgs_ids
        }
        if imgs_ids
        else {}
    )
    imgs_info_subset = list(subset.values()) if imgs_ids else imgs_info
    indices = list(subset.keys())

    # Define a dataset that is a subset of the initial dataset
    dataset_subset = Subset(dataset, indices) if imgs_ids else dataset

    return {"images_info": imgs_info_subset, "annotations": dataset_subset}


def convert_predictions_to_coco_format(
    imgs_info: list[dict[str, Any]],
    preds: list[torch.Tensor],
    threshold: float = 0.0,
    output_stride_h: int = 4,
    output_stride_w: int = 4,
    output_file: str = None,
) -> list[dict[str, object]]:
    """
    Converts model predictions into COCO format.

    This function processes a list of predicted tensors and image metadata, converting
    the predictions into a COCO-compatible format. Each prediction includes the image ID,
    category ID, bounding box coordinates, and confidence score.

    Args:
        imgs_info (list[dict[str, Any]]): A list of dictionaries containing image metadata,
            including image width, height, and ID.
        preds (list[torch.Tensor]): A list of tensors representing model predictions. Each tensor
            has a shape of (num_categories + 4, H, W), where the last 4 channels represent the bounding box.
        threshold (float, optional): The confidence threshold for filtering predictions. Defaults to 0.0.
        output_stride_h (int, optional): The stride of the output feature map in height. Defaults to 4.
        output_stride_w (int, optional): The stride of the output feature map in width. Defaults to 4.
        output_file (str, optional): Path to save the results as a JSON file. If None, the results
            are not saved to a file. Defaults to None.

    Returns:
        list[dict[str, object]]: A list of dictionaries formatted according to COCO annotation format.
            The list has the following form:

            .. code-block:: python

                [
                    {
                        "image_id": int,                # The ID of the corresponding image.
                        "category_id": int,             # The ID of the detected category.
                        "bbox": [x, y, width, height],  # The bounding box in COCO format [x, y, width, height].
                        "score": float,                 # The confidence score of the prediction.
                    },
                    ...,
                ]

    Notes:
        - The function rescales bounding boxes based on the original image dimensions.
        - It uses a progress bar (via `tqdm`) to track processing progress.
        - If `output_file` is provided, the results are saved as a JSON file.
    """
    results = []

    pred_shape = preds[0].shape
    num_categories = pred_shape[0] - 4

    total = len(imgs_info)

    with tqdm(total=total) as pbar:
        for img_info, pred in zip(imgs_info, preds):
            width_scale_factor = float(img_info["width"] / IMG_WIDTH)
            height_scale_factor = float(img_info["height"] / IMG_HEIGHT)

            for category in range(num_categories):
                for i in range(pred_shape[1]):
                    for j in range(pred_shape[2]):
                        box = pred[num_categories:, i, j].tolist()
                        score = pred[category, i, j].item()
                        if score >= threshold:
                            results.append(
                                {
                                    "image_id": img_info["id"],
                                    "category_id": category + 1,
                                    "bbox": [
                                        (j * output_stride_h - box[0])
                                        * width_scale_factor,
                                        (i * output_stride_w - box[1])
                                        * height_scale_factor,
                                        (box[2] + box[0]) * width_scale_factor,
                                        (box[3] + box[1]) * height_scale_factor,
                                    ],
                                    "score": score,
                                }
                            )
            pbar.update(1)

    if output_file:
        print(f"Storing result in file: {output_file}...")
        with open(output_file, "w") as f:
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

    parser = argparse.ArgumentParser(
        description="This script converts predictions to coco format json"
    )

    parser.add_argument("imgs_dir", type=str, help="path to images")

    parser.add_argument(
        "ann_file", type=str, help="path to json-file with annotation in COCO format"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="threshold for filtering predictions with low confidence",
    )

    parser.add_argument(
        "--imgs_ids",
        type=str,
        default="",
        help="list of images ids to get predictions; "
        "should be passed in form 1,2,3,4,5",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="output file to store converted predictions",
    )

    args = parser.parse_args()

    imgs_ids = args.imgs_ids.split(",")
    imgs_ids_int = list(map(int, imgs_ids))

    model = load_model(device, ModelBuilder, alpha=0.25)

    dataset = prepare_dataset(
        imgs_dir=args.imgs_dir,
        ann_file=args.ann_file,
        imgs_ids=imgs_ids_int,
    )
    dataset_transformed = transform_dataset(dataset["annotations"])
    predictions = get_predictions(device, model, dataset_transformed)
    predictions = [pred.squeeze() for pred in predictions]

    num_categories = predictions[0].shape[0] - BBOX_PART_LEN

    _ = convert_predictions_to_coco_format(
        dataset["images_info"],
        predictions,
        threshold=args.threshold,
        output_file=args.output_file or None,
    )
