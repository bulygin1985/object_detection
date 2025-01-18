"""
Convert VOC format dataset to COCO.
Reference 1: https://github.com/autogluon/autogluon/blob/master/multimodal/src/autogluon/multimodal/utils/object_detection.py
Reference 2: https://github.com/yukkyo/voc2coco/blob/master/voc2coco.py
1. id stored as int
2. provide only root_dir, and corresponding simplification
3. Use defusedxml.ElementTree for security concern
4. remove invalid images (without bounding boxes or too small bounding boxes)
5. this script doesn't convert segmentation

Example to run:
python voc2coco.py --voc_ann_dir /path/to/input/voc/Annotations --coco_ann_dir /path/to/output/coco/annotations --labels_file /path/to/output/labels.txt --min_area 4 --output_form both --voc_ann_ids_dir /path/to/input/voc/ImageSets/Main

or using the default values:

python voc2coco.py --voc_ann_dir /path/to/input/voc/Annotations
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import defusedxml.ElementTree as ET
from tqdm import tqdm

ALLOWED_EXTENSIONS = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]

JOINED = "joined"
BOTH = "both"
SPLIT = "split"


def dump_voc_classes(
    voc_annotation_path: str, voc_class_names_output_path: str = None
) -> [str]:
    """
    Reads annotations for a dataset in VOC format.
    Then
        dumps the unique class names into a labels.txt file.
    Parameters
    ----------
    voc_annotation_path
        root_path for annotations in VOC format
    voc_class_names_output_path
        output path for the labels.txt
    Returns
    -------
    list of strings, [class_name0, class_name1, ...]
    """
    files = os.listdir(voc_annotation_path)
    class_names = set()
    for f in files:
        if f.endswith(".xml"):
            xml_path = os.path.join(voc_annotation_path, f)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for boxes in root.iter("object"):
                class_names.add(boxes.find("name").text)

    sorted_class_names = sorted(list(class_names))
    if voc_class_names_output_path:
        with open(voc_class_names_output_path, "w") as f:
            f.writelines("\n".join(sorted_class_names))

    return sorted_class_names


def get_class2id(classes: List[str]) -> Dict[str, int]:
    """id is 1 start"""
    classes_ids = list(range(1, len(classes) + 1))
    return dict(zip(classes, classes_ids))


def get_ann_files(ann_ids_dir: str) -> Dict:
    ann_paths = {}
    for ann_ids_filename in os.listdir(ann_ids_dir):
        ann_ids_path = os.path.join(ann_ids_dir, ann_ids_filename)

        filename, ext = os.path.splitext(ann_ids_filename)
        if os.path.isfile(ann_ids_path) and ext == ".txt":
            ann_ids_name = filename

            with open(ann_ids_path, "r") as f:
                rows = f.readlines()
                if not rows:
                    # todo (AA): log
                    print(f"Skipping {ann_ids_path}: file is empty")
                else:
                    ann_ids = []
                    for r in rows:
                        data = r.strip().split()
                        if len(data) == 1:  # Each row is an annotation id
                            ann_ids.append(data[0])
                        elif (
                            len(data) == 2
                        ):  # Each row contains an annotation id and a flag (0 if we do not use this annotation in this split, and 1 if we use it)
                            ann_id, used = data
                            if int(used) == 1:
                                ann_ids.append(ann_id)
                        else:
                            # todo (AA): log error
                            print(
                                f"Skipping {ann_ids_path}: file format not recognized. Make sure your annotation follows "
                                f"VOC format!"
                            )
                            break

                    ann_paths[ann_ids_name] = [aid + ".xml" for aid in ann_ids]
    return ann_paths


def get_image_info(annotation_root):
    path = annotation_root.findtext("path")
    if path is None:
        filename = annotation_root.findtext("filename")
    else:
        filename = os.path.basename(path)
    img_name = os.path.basename(filename)

    img_id, img_ext = os.path.splitext(img_name)
    if not img_ext in ALLOWED_EXTENSIONS:
        raise ValueError(f"Image extension is not valid! Got {img_name}")

    if not img_id.isdigit():
        raise ValueError(f"Image file should contain digits only! Got {img_name}")

    size = annotation_root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    image_info = {
        "file_name": img_name,
        "height": height,
        "width": width,
        "id": int(img_id),
    }
    return image_info


def get_coco_annotation_from_obj(obj, label2id, min_area):
    label = obj.findtext("name")
    assert label in label2id, f"Error: {label} is not in label2id!"
    category_id = label2id[label]
    bndbox = obj.find("bndbox")
    xmin = int(float(bndbox.findtext("xmin")))
    ymin = int(float(bndbox.findtext("ymin")))
    xmax = int(float(bndbox.findtext("xmax")))
    ymax = int(float(bndbox.findtext("ymax")))
    if xmin >= xmax or ymin >= ymax:
        return {}
    o_width = xmax - xmin
    o_height = ymax - ymin
    area = o_width * o_height
    if area <= min_area:
        return {}
    ann = {
        "area": o_width * o_height,
        "iscrowd": 0,
        "bbox": [xmin, ymin, o_width, o_height],
        "category_id": category_id,
        "ignore": 0,
        "segmentation": [],  # This script is not for segmentation
    }
    return ann


def convert_xmls_to_cocojson(
    voc_ann_dir: str,
    annotation_files: List[str],
    label2id: Dict[str, int],
    output_jsonpath: str,
    min_area: int,
):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": [],
    }
    bnd_id = 1  # START_BOUNDING_BOX_ID
    print("Start converting!")
    for a_file in tqdm(annotation_files):
        # Read annotation xml
        ann_tree = ET.parse(os.path.join(voc_ann_dir, a_file))
        ann_root = ann_tree.getroot()

        try:
            img_info = get_image_info(annotation_root=ann_root)
        except ValueError as e:
            # todo (AA): log error
            print(f"Cannot get image info due to the error: {e}")
            continue
        img_id = img_info["id"]

        valid_image = (
            False  # remove image without bounding box to speed up mAP calculation
        )
        for obj in ann_root.findall("object"):
            ann = get_coco_annotation_from_obj(
                obj=obj, label2id=label2id, min_area=min_area
            )
            if ann:
                ann.update({"image_id": img_id, "id": bnd_id})
                output_json_dict["annotations"].append(ann)
                bnd_id = bnd_id + 1
                valid_image = True

        if valid_image:
            output_json_dict["images"].append(img_info)
        else:
            print(
                f"Image {img_id} is removed since it does not contain any valid object"
            )

    for label, label_id in label2id.items():
        category_info = {"supercategory": "none", "id": label_id, "name": label}
        output_json_dict["categories"].append(category_info)

    with open(output_jsonpath, "w") as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)
        print(f"The COCO format annotation is saved to {output_jsonpath}")


def main():
    parser = argparse.ArgumentParser(
        description="This script converts voc format xmls to coco format json"
    )
    parser.add_argument(
        "voc_ann_dir", type=str, help="path to annotations in VOC format"
    )
    parser.add_argument(
        "--voc_ann_ids_dir",
        type=str,
        default="",
        help="path to VOC annotation ids files, if not provided, path relative to voc_ann_dir "
        "will be inferred according to the standard file structure of VOC dataset",
    )
    parser.add_argument(
        "--coco_ann_dir",
        type=str,
        default="",
        help="path to output annotations in COCO format",
    )
    parser.add_argument(
        "--labels_file", type=str, default="", help="path to output file with labels"
    )
    parser.add_argument(
        "--min_area", type=str, default=4, help="min area for a valid bounding box"
    )
    parser.add_argument(
        "--output_form",
        type=str,
        default="joined",
        choices=["split", "joined", "both"],
        help="should we split output annotations by category, join all into one, or generate both?",
    )

    args = parser.parse_args()

    min_area = args.min_area
    assert min_area >= 0

    output_form = args.output_form

    voc_ann_dir = args.voc_ann_dir
    coco_ann_dir = args.coco_ann_dir or voc_ann_dir

    voc_ann_ids_dir = args.voc_ann_ids_dir
    if not voc_ann_ids_dir:
        voc_ann_ids_dir = str(Path(voc_ann_dir).parent / "ImageSets" / "Main")

    labels_file = args.labels_file

    # generate labels.txt containing all unique class names
    classes = dump_voc_classes(
        voc_annotation_path=voc_ann_dir, voc_class_names_output_path=labels_file
    )

    label2id = get_class2id(classes=classes)

    if output_form in [SPLIT, BOTH]:
        output_path_fmt = os.path.join(coco_ann_dir, "%s_cocoformat.json")
        ann_files = get_ann_files(ann_ids_dir=voc_ann_ids_dir)
        for mode, ann_file in ann_files.items():
            convert_xmls_to_cocojson(
                voc_ann_dir=voc_ann_dir,
                annotation_files=ann_file,
                label2id=label2id,
                output_jsonpath=output_path_fmt % mode,
                min_area=min_area,
            )

    if output_form in [JOINED, BOTH]:
        ann_files = [
            ann_filename
            for ann_filename in os.listdir(voc_ann_dir)
            if os.path.isfile(os.path.join(voc_ann_dir, ann_filename))
            and os.path.splitext(ann_filename)[1] == ".xml"
        ]
        convert_xmls_to_cocojson(
            voc_ann_dir=voc_ann_dir,
            annotation_files=ann_files,
            label2id=label2id,
            output_jsonpath=os.path.join(coco_ann_dir, "all_cocoformat.json"),
            min_area=min_area,
        )


if __name__ == "__main__":
    main()