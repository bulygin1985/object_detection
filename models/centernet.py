import torch.nn as nn

from losses.centernet_ttf import CenternetTTFLoss
from models.backbones import create_backbone
from models.centernet_head import Head

# todo (AA): move it somewhere
input_height = input_width = 256


class ModelBuilder(nn.Module):
    """
    To connect head with backbone
    """

    def __init__(
        self,
        alpha=1.0,
        class_number=20,
        backbone: str = "default",
        backbone_weights: str = None,
    ):
        super().__init__()
        self.class_number = class_number
        self.backbone = create_backbone(backbone, alpha, backbone_weights)
        self.head = Head(
            backbone_output_filters=self.backbone.filters, class_number=class_number
        )
        self.loss = CenternetTTFLoss(
            # todo (AA): is this "4" below the down_ratio parameter?
            #   shouldn't we pass it as an argument to initializer?
            #   shouldn't we pass input_height and input_width as arguments too?
            class_number,
            4,
            input_height // 4,
            input_width // 4,
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
