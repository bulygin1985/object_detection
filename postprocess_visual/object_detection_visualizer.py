import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from postprocess_visual.postprocess import CenternetPostprocess
from postprocess_visual.visualizer import PASCAL_CLASSES


class ObjectDetectionVisualizer:
    def __init__(
        self,
        dataset,
        input_height=256,
        input_width=256,
        down_ratio=4,
        confidence_threshold=0.3,
    ):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        self.input_height = input_height
        self.input_width = input_width
        self.down_ratio = down_ratio
        self.confidence_threshold = confidence_threshold

        self._setup()

    def _setup(self):
        try:
            self.postprocessor = CenternetPostprocess(
                n_classes=20,
                width=self.input_width,
                height=self.input_height,
                down_ratio=self.down_ratio,
            ).to(self.device)
        except Exception as e:
            self.logger.error(f"Error setting up components: {e}")
            raise

    def _process_detections(self, detections, img_h, img_w):
        pred_boxes = []
        pred_labels = []
        pred_scores = []

        for det in detections[0]:
            class_id = int(det[0].item())
            score = det[1].item()

            if score <= self.confidence_threshold:
                continue

            x1 = int(det[2].item() * img_w)
            y1 = int(det[3].item() * img_h)
            x2 = int(det[4].item() * img_w)
            y2 = int(det[5].item() * img_h)

            pred_boxes.append([x1, y1, x2, y2])
            pred_labels.append(class_id)
            pred_scores.append(score)

        return pred_boxes, pred_labels, pred_scores

    def _get_heatmap_visualization(self, heatmaps):
        max_heatmap, _ = torch.max(heatmaps[0, :20], dim=0)
        heatmap_np = max_heatmap.cpu().numpy()
        heatmap_np = (heatmap_np - heatmap_np.min()) / (
            heatmap_np.max() - heatmap_np.min() + 1e-8
        )

        colored_heatmap = cv2.applyColorMap(
            (heatmap_np * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        return cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    def _plot_detection_results(
        self,
        orig_img,
        colored_heatmap,
        img_with_predictions,
        pred_scores,
        sample_index,
    ):
        _ = plt.figure(figsize=(20, 5))

        plt.subplot(1, 4, 1)
        plt.title(f"Original Image {sample_index + 1}")
        plt.imshow(orig_img)
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.title(f"Heatmap {sample_index + 1}")
        plt.imshow(colored_heatmap)
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.title(f"Predictions {sample_index + 1}")
        plt.imshow(img_with_predictions)
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.title(f"Detection Scores {sample_index + 1}")
        if pred_scores:
            y_pos = np.arange(len(pred_scores))
            plt.barh(y_pos, pred_scores)
            plt.yticks(y_pos, [f"Det {i + 1}" for i in range(len(pred_scores))])
            plt.xlabel("Confidence Score")
        else:
            plt.text(0.5, 0.5, "No detections", ha="center", va="center")

    def visualize_predictions(self, preds):
        for i, (orig_img, _) in enumerate(self.dataset):
            pred = preds[i]

            heatmaps = pred[:, :20]
            colored_heatmap = self._get_heatmap_visualization(heatmaps)
            detections = self.postprocessor(pred)

            img_np = np.asarray(orig_img).copy()

            pred_boxes, pred_labels, pred_scores = self._process_detections(
                detections, self.input_height, self.input_width
            )

            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                cv2.rectangle(
                    img_np,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    (0, 255, 0),
                    2,
                )
                label_text = f"{PASCAL_CLASSES[label - 1]}: {score:.2f}"
                cv2.putText(
                    img_np,
                    label_text,
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            self._plot_detection_results(
                orig_img, colored_heatmap, img_np, pred_scores, i
            )

            plt.tight_layout()
            plt.show()
