import json
from numpy import array
from time import time
from typing import Literal

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate(
        model_predictions: str = None,
        model_predictions_object: dict = None,
        ground_truth_annotations: str = None,
        ground_truth_object: dict = None,
        img_ids=None,
        cat_ids=None,
        ann_type: Literal["segm", "bbox", "keypoints"] = "bbox"
):
    assert not(model_predictions and model_predictions_object), \
        "Error: only one of 'model_predictions' and 'model_predictions_object' may be provided"
    assert (model_predictions or model_predictions_object), \
        "Error: one of 'model_predictions' and 'model_predictions_object' must be provided"
    assert not(ground_truth_annotations and ground_truth_object), \
        "Error: only one of 'ground_truth_annotations' and 'ground_truth_object' may be provided"
    assert (ground_truth_annotations or ground_truth_object), \
        "Error: one of 'ground_truth_annotations' and 'ground_truth_object' must be provided"

    if ground_truth_annotations:
        print('loading annotations into memory...')
        tic = time()
        with open(ground_truth_annotations, 'r') as f:
            ground_truth_object = json.load(f)
        assert type(ground_truth_object) == dict, (
            'annotation file format {} not supported'.format(type(ground_truth_object)))
        print('Done (t={:0.2f}s)'.format(time() - tic))

    # todo (AA): filter ground_truth_object by img_ids and cat_ids

    coco_gt = COCO()
    coco_gt.dataset = ground_truth_object
    coco_gt.createIndex()

    if model_predictions:
        with open(model_predictions) as f:
            model_predictions_object = json.load(f)

    # todo (AA): filter model_predictions_object by img_ids and cat_ids

    model_predictions_list = [
        [
            elem["image_id"],
            elem["bbox"][0],
            elem["bbox"][1],
            elem["bbox"][2],
            elem["bbox"][3],
            elem["score"],
            elem["category_id"],
        ] for elem in model_predictions_object
    ]
    coco_dt = coco_gt.loadRes(array(model_predictions_list))

    filtered_img_ids = sorted(coco_gt.getImgIds(imgIds=img_ids or [], catIds=cat_ids or []))

    coco_eval = COCOeval(coco_gt, coco_dt, ann_type)
    coco_eval.params.imgIds = filtered_img_ids

    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats


def __main__():
    _ = evaluate(
        ground_truth_annotations="../VOC_COCO/pascal_trainval2007.json",
        model_predictions="data/predictions/predictions.json"
    )
