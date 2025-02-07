import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

PASCAL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def print_text(img, text, x1, y1, color):
    fontScale = 0.4
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, 1)
    img = cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), color, -1)
    img = cv2.putText(
        img=img,
        text=text,
        org=(x1, y1 + h),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=fontScale,
        color=(255, 255, 255),
        thickness=1,
        bottomLeftOrigin=False,
    )


def draw_bbox(image, box, category_name, color=(255, 0, 0), line_width=2):
    x_min, y_min, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
    cv2.rectangle(
        img=image,
        rec=(int(x_min), int(y_min), int(width), int(height)),
        color=color,
        thickness=line_width,
    )
    print_text(img=image, text=category_name, x1=int(x_min), y1=int(y_min), color=color)


def get_image_with_bboxes(img, boxes, labels):
    img_np = np.array(img)
    for cls_id, box in zip(labels.data.numpy(), boxes.data.numpy()):
        draw_bbox(image=img_np, category_name=PASCAL_CLASSES[cls_id - 1], box=box)
    return img_np


def visualize_heatmap(heatmap):
    """
    Visualize a single heatmap
    Args:
        heatmap (torch.Tensor): Single channel heatmap tensor
        title (str): Title for the plot
    Returns:
        np.ndarray: Colored heatmap image
    """
    heatmap_np = heatmap.cpu().numpy()
    heatmap_np = np.maximum(heatmap_np, 0)
    heatmap_np = (
        heatmap_np / np.max(heatmap_np) if np.max(heatmap_np) > 0 else heatmap_np
    )

    colored_heatmap = cv2.applyColorMap(
        (heatmap_np * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    return colored_heatmap
