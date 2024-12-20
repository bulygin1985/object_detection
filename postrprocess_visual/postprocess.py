import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CenternetPostprocess(nn.Module):

    def __init__(
        self,
        n_classes: int = 80,
        width: int = 320,
        height: int = 320,
        down_ratio: int = 4,
        max_detections: int = 100,
    ):
        """
        Initialize CenterNet postprocessing layer.

        Args:
            n_classes (int): Number of object classes
            width (int): Input image width
            height (int): Input image height
            down_ratio (int): Downsampling ratio of the network
            max_detections (int): Maximum number of detections per image
        """
        super().__init__()
        self._n_classes = n_classes
        self._down_ratio = down_ratio
        self._width = width
        self._height = height
        self._max_detections = max_detections
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, y_pred: torch.Tensor) -> torch.Tensor:

        hm = y_pred[:, : self._n_classes, :, :]
        coors = y_pred[:, self._n_classes : self._n_classes + 4, :, :]

        return self.decode_center_detections(
            hm,
            coors,
            output_stride=self._down_ratio,
            max_detections=self._max_detections,
        )

    def _non_maximum_suppression(
        self, heatmap: torch.Tensor, kernel: int = 3
    ) -> torch.Tensor:
        """
        Args:
            heatmap (torch.Tensor): Input heatmap
            kernel (int): Suppression kernel size
        Returns:
            torch.Tensor: Suppressed heatmap
        """
        hmax = F.max_pool2d(heatmap, kernel, stride=1, padding=kernel // 2)
        keep = torch.eq(hmax, heatmap).type(torch.float32).to(self.device)
        return heatmap.type(torch.float32) * keep

    def decode_center_detections(
        self,
        heatmaps: torch.Tensor,
        coordinates: torch.Tensor,
        output_stride: int,
        max_detections: int,
    ) -> torch.Tensor:
        """
        Decode center point detections from heatmaps and coordinate predictions.

        Args:
            heatmaps (torch.Tensor): Class heatmaps
            coordinates (torch.Tensor): Coordinate offsets
            output_stride (int): Ratio between detector input and output image sizes
            max_detections (int): Maximum number of detections per image

        Returns:
            torch.Tensor: Detected objects with [class_id, score, x_min, y_min, x_max, y_max]
        """
        heatmaps = self._non_maximum_suppression(heatmaps)

        heatmaps = heatmaps.permute(0, 2, 3, 1)
        coordinates = coordinates.permute(0, 2, 3, 1)

        batch_size, width, num_classes = (
            heatmaps.shape[0],
            heatmaps.shape[2],
            heatmaps.shape[3],
        )

        heatmaps_flat = torch.reshape(heatmaps, (batch_size, -1))
        coordinates_flat = torch.reshape(
            coordinates, (coordinates.shape[0], -1, coordinates.shape[-1])
        )

        # Extract top detections
        detections = self._extract_top_detections(
            heatmaps_flat,
            coordinates_flat,
            width,
            num_classes,
            output_stride,
            max_detections,
        )

        return detections

    def _extract_top_detections(
        self,
        heatmaps_flat: torch.Tensor,
        coordinates_flat: torch.Tensor,
        width: int,
        num_classes: int,
        output_stride: int,
        max_detections: int,
    ) -> torch.Tensor:
        """
        Extract top-k detections from flattened heatmaps and coordinates.

        Args:
            heatmaps_flat (torch.Tensor): Flattened heatmaps
            coordinates_flat (torch.Tensor): Flattened coordinate offsets
            width (int): Width of feature map
            num_classes (int): Number of classes
            output_stride (int): Network output stride
            max_detections (int): Maximum number of detections

        Returns:
            torch.Tensor: Top-k detections with class, score, and bbox coordinates
        """
        # Get top-k scores and indices
        scores, indices = torch.topk(
            heatmaps_flat, k=max_detections, dim=1, sorted=True
        )

        # Extract classes
        classes = (indices % num_classes).type(torch.float32) + 1.0
        indices = (indices / num_classes).type(torch.int64)

        # Calculate x, y coordinates
        xs = (indices % width).type(torch.float32)
        ys = (indices / width).type(torch.int32).type(torch.float32)

        # Calculate bounding box coordinates
        x1 = (
            output_stride * xs
            - coordinates_flat[..., 0].gather(dim=1, index=indices).type(torch.float32)
        ) / self._width

        y1 = (
            output_stride * ys
            - coordinates_flat[..., 1].gather(dim=1, index=indices).type(torch.float32)
        ) / self._height

        x2 = (
            output_stride * xs
            + coordinates_flat[..., 2].gather(dim=1, index=indices).type(torch.float32)
        ) / self._width

        y2 = (
            output_stride * ys
            + coordinates_flat[..., 3].gather(dim=1, index=indices).type(torch.float32)
        ) / self._height

        detections = torch.stack([classes, scores, x1, y1, x2, y2, ys, xs], dim=-1)

        return detections
