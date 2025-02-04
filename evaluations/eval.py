import json
from collections import defaultdict
from time import time
from typing import Literal

from numpy import array
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.transforms import v2 as v2_transforms

IMG_HEIGHT = IMG_WIDTH = 256


class MAPEvaluator:
    def __init__(
        self,
        model_predictions: str = None,
        model_predictions_object: dict = None,
        ground_truth_annotations: str = None,
        ground_truth_object: dict = None,
        ann_type: Literal["segm", "bbox", "keypoints"] = "bbox",
    ):
        # validate passed parameters
        assert not (
            model_predictions and model_predictions_object
        ), "Error: only one of 'model_predictions' and 'model_predictions_object' may be provided"
        assert (
            model_predictions or model_predictions_object
        ), "Error: one of 'model_predictions' and 'model_predictions_object' must be provided"
        assert not (
            ground_truth_annotations and ground_truth_object
        ), "Error: only one of 'ground_truth_annotations' and 'ground_truth_object' may be provided"
        assert (
            ground_truth_annotations or ground_truth_object
        ), "Error: one of 'ground_truth_annotations' and 'ground_truth_object' must be provided"

        # load ground truth annotations
        self.ground_truth_object = ground_truth_object
        if ground_truth_annotations:
            print("loading annotations into memory...")
            tic = time()
            with open(ground_truth_annotations, "r") as f:
                self.ground_truth_object = json.load(f)
            assert (
                type(self.ground_truth_object) == dict
            ), "annotation file format {} not supported".format(
                type(self.ground_truth_object)
            )
            print("Done (t={:0.2f}s)".format(time() - tic))

        # load model predictions
        self.model_predictions_object = model_predictions_object
        if model_predictions:
            with open(model_predictions) as f:
                self.model_predictions_object = json.load(f)

        self.ground_truth_filtered = self.ground_truth_object
        self.model_predictions_filtered = self.model_predictions_object

        self.img_ids = []
        self.cat_ids = []
        self.ann_type = ann_type

        self.coco_eval = None

    def filter_input(self, img_ids: list = None, cat_ids: list = None):
        """Filter ground_truth_object by img_ids and cat_ids"""
        if img_ids:
            self.img_ids = img_ids

            self.ground_truth_filtered["annotations"] = [
                elem
                for elem in self.ground_truth_object.get("annotations", [])
                if elem["image_id"] in img_ids
            ]
            self.ground_truth_filtered["images"] = [
                elem
                for elem in self.ground_truth_object.get("images", [])
                if elem["id"] in img_ids
            ]

        if cat_ids:
            self.cat_ids = cat_ids

            new_img_ids = set(
                elem["id"] for elem in self.ground_truth_filtered.get("images", [])
            )

            cat_to_imgs = defaultdict(list)
            for ann in self.ground_truth_filtered.get("annotations", []):
                cat_to_imgs[ann["category_id"]].append(ann["image_id"])

            for i, cat_id in enumerate(cat_ids):
                new_img_ids &= set(cat_to_imgs[cat_id])

            self.ground_truth_filtered["annotations"] = [
                elem
                for elem in self.ground_truth_filtered.get("annotations", [])
                if elem["image_id"] in new_img_ids
            ]
            self.ground_truth_filtered["images"] = [
                elem
                for elem in self.ground_truth_filtered.get("images", [])
                if elem["id"] in new_img_ids
            ]

        # filter model_predictions_object by img_ids and cat_ids
        if img_ids:
            self.model_predictions_filtered = [
                elem
                for elem in self.model_predictions_object
                if elem["image_id"] in img_ids
            ]
        if cat_ids:
            self.model_predictions_filtered = [
                elem
                for elem in self.model_predictions_filtered
                if elem.get("category_id", []) in cat_ids
            ]

    def evaluate(self) -> list[dict]:
        model_predictions = array(
            [
                [
                    elem["image_id"],
                    elem["bbox"][0],
                    elem["bbox"][1],
                    elem["bbox"][2],
                    elem["bbox"][3],
                    elem["score"],
                    elem["category_id"],
                ]
                for elem in self.model_predictions_filtered
            ]
        )

        coco_gt = COCO()
        coco_gt.dataset = self.ground_truth_filtered
        coco_gt.createIndex()

        coco_dt = coco_gt.loadRes(model_predictions)

        coco_eval = COCOeval(coco_gt, coco_dt, self.ann_type)

        # Run evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        self.coco_eval = coco_eval
        return coco_eval.stats


if __name__ == "__main__":
    evaluator = MAPEvaluator(
        ground_truth_annotations="../VOC_COCO/pascal_trainval2007.json",
        model_predictions="../VOC_COCO/pascal_train2007_predictions.json",
    )

    # the list [12, 17, 23, 26, 32, 33, 34, 35, 36, 42] contains first 10 image ids of train pascal VOC dataset
    evaluator.filter_input(
        img_ids=[12, 17, 23, 26, 32, 33, 34, 35, 36, 42], cat_ids=None
    )

    _ = evaluator.evaluate()
