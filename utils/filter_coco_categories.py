import json
import os
import sys
from typing import Dict, List


def filter_dict_array(data: List[Dict], key: str, categories: List[int]):
    new_data = [d.copy() for d in data if d[key] in categories]
    for d in new_data:
        new_category_id = categories.index(d[key]) + 1
        d[key] = new_category_id
    return new_data


def filter_coco_annotation_dict(data: Dict, categories: List[int]):
    """Filter coco annotations data dict with specified categories
    Args:
        data (dict): dictionary of coco annotations
        categories (list[int]): list of categories to keep
    Returns:
        dict with only only specified categories in annotations and categories.
    """
    result = data.copy()
    result["annotations"] = filter_dict_array(
        data["annotations"], "category_id", categories
    )
    result["categories"] = filter_dict_array(data["categories"], "id", categories)
    return result


def main():
    """Filter coco annotations json with specified categories.
    Filter one class examples:
         python filter_coco_categies.py pascal_val2007.json 15
    Filter 3 classes examples:
         python filter_coco_categies.py pascal_val2007.json 10,12,15
    """
    if len(sys.argv) != 3:
        print(
            "filter_coco_categies.\nUsage:\n\tpython filter_coco_categies.py <input_filename> <categories_list>"
        )
        return
    fname_input = sys.argv[1]
    categories_str = sys.argv[2].split(",")
    categories = list(map(int, categories_str))
    head, ext = os.path.splitext(fname_input)
    file_input = open(fname_input, "r")
    data = json.load(file_input)
    data_filtered = filter_coco_annotation_dict(data, categories)
    fname_output = head + "_filtered_" + "_".join(categories_str) + ext
    file_output = open(fname_output, "w")
    json.dump(data_filtered, file_output)
    print(f"filtered file created as {fname_output}")


if __name__ == "__main__":
    main()
