import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import Subset
from torchvision.transforms import v2 as transforms

from models.centernet import ModelBuilder
from postrprocess_visual.postprocess import CenternetPostprocess
from postrprocess_visual.visualizer import PASCAL_CLASSES
from training.encoder import CenternetEncoder


class ObjectDetectionVisualizer:
    def __init__(
        self,
        input_height=256,
        input_width=256,
        down_ratio=4,
        checkpoint_path=None,
        confidence_threshold=0.3,
    ):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        self.input_height = input_height
        self.input_width = input_width
        self.down_ratio = down_ratio
        self.confidence_threshold = confidence_threshold

        self.checkpoint_path = (
            "../models/checkpoints/pretrained_weights.pt"
            if checkpoint_path is None
            else checkpoint_path
        )

        self._setup()

    def _setup(self):
        try:
            self.dataset = self._prepare_dataset()
            self.transform = self._create_transforms()
            self.encoder = CenternetEncoder(self.input_height, self.input_width)
            self.postprocessor = CenternetPostprocess(
                n_classes=20,
                width=self.input_width,
                height=self.input_height,
                down_ratio=self.down_ratio,
            ).to(self.device)
            self.model = self._load_trained_model()
        except Exception as e:
            self.logger.error(f"Error setting up components: {e}")
            raise

    def _prepare_dataset(self):
        try:
            dataset_val = torchvision.datasets.VOCDetection(
                root="../VOC", year="2007", image_set="val", download=True
            )
            dataset_val = torchvision.datasets.wrap_dataset_for_transforms_v2(
                dataset_val
            )
            indices = list(range(min(10, len(dataset_val))))
            dataset_val = Subset(dataset_val, indices)
            return dataset_val
        except Exception as e:
            self.logger.error(f"Error preparing dataset: {e}")
            raise

    def _create_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize(size=(self.input_width, self.input_height)),
                transforms.ToTensor(),
            ]
        )

    def _load_trained_model(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {self.checkpoint_path}"
            )

        model = ModelBuilder(alpha=0.25).to(self.device)
        model.load_state_dict(
            torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=True
            )
        )
        model.eval()
        return model

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
        heatmap_resized = cv2.resize(
            heatmap_np,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        colored_heatmap = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        return cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    def _plot_detection_results(
        self, orig_img, colored_heatmap, img_with_predictions, pred_scores, sample_index
    ):
        fig = plt.figure(figsize=(20, 5))

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
        plt.tight_layout()

    def visualize_predictions(self, num_samples=5):
        self.num_samples = min(num_samples, len(self.dataset))

        for i in range(self.num_samples):
            orig_img, orig_label = self.dataset[i]
            img, bboxes, labels = self.transform(
                orig_img, orig_label["boxes"], orig_label["labels"]
            )
            img = img.unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred = self.model(img)

            heatmaps = pred[:, :20]
            colored_heatmap = self._get_heatmap_visualization(heatmaps)
            detections = self.postprocessor(pred)

            img_np = np.transpose(img.cpu().squeeze().numpy(), (1, 2, 0))
            img_np = np.clip(img_np, 0, 1)
            img_with_predictions = img_np.copy()

            pred_boxes, pred_labels, pred_scores = self._process_detections(
                detections, self.input_height, self.input_width
            )

            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                cv2.rectangle(
                    img_with_predictions,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    (0, 1, 0),
                    2,
                )
                label_text = f"{PASCAL_CLASSES[label - 1]}: {score:.2f}"
                cv2.putText(
                    img_with_predictions,
                    label_text,
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 1, 0),
                    2,
                )

            self._plot_detection_results(
                orig_img, colored_heatmap, img_with_predictions, pred_scores, i
            )
            plt.show()
