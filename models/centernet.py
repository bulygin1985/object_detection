import torch.nn as nn

from losses.centernet_ttf import CenternetTTFLoss
from models.backbones import create_backbone
from models.centernet_head import Head
from utils.config import IMG_HEIGHT, IMG_WIDTH


class ModelBuilder(nn.Module):
    """
    To connect head with backbone
    """

    def __init__(
        self,
        filters_size: list = None,
        alpha=1.0,
        class_number=20,
        backbone: str = "default",
        backbone_weights: str = None,
    ):
        super().__init__()
        self.class_number = class_number
        self.backbone = create_backbone(backbone, alpha, backbone_weights)
        if not filters_size:
            filters_size = [128, 64, 32]
        self.head = Head(
            backbone_output_filters=self.backbone.filters,
            filters_size=filters_size,
            class_number=class_number,
        )
        self.loss = CenternetTTFLoss(
            # todo (AA): is this "4" below the down_ratio parameter?
            #   shouldn't we pass it as an argument to initializer?
            #   shouldn't we pass input_height and input_width as arguments too?
            class_number,
            4,
            IMG_HEIGHT // 4,
            IMG_WIDTH // 4,
        )

    def forward(self, x, gt=None):
        x = x / 0.5 - 1.0  # normalization
        out = self.backbone(x)
        pred = self.head(*out)

        if gt is None:
            return pred
        else:
            loss = self.loss(gt, pred)
            return loss
