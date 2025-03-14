import json
import os
import sys


def filter_coco_annotation_dict(data, categories):
    annotations = data["annotations"]
    new_annotations = [a.copy() for a in annotations if a["category_id"] in categories]
    for a in new_annotations:
        new_category_id = categories.index(a["category_id"]) + 1
        a["category_id"] = new_category_id
    result = data.copy()
    result["annotations"] = new_annotations
    return result


def main():
    if len(sys.argv) <= 1:
        print(
            "filter_coco_categies. Usage:\n python filter_coco_categies.py <input_filename> <categories_list>"
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
