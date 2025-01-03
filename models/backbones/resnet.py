import torch.nn as nn
import torchvision.models as models

from .abstract_backbone import AbstractBackbone


def count_filters(resnet_layer: nn.Sequential) -> int:
    """Get number of output filters (channels) for resnet layer
    Args:
        resnet_layer (nn.Sequential): resnet model layer (one out of 4 layer1..layer4).

    Returns:
        int: number of output filters for the input layer.
    """
    lastblock = resnet_layer[-1]
    if isinstance(lastblock, models.resnet.BasicBlock):
        filters = lastblock.conv2.weight.shape[0]
    elif isinstance(lastblock, models.resnet.Bottleneck):
        filters = lastblock.conv3.weight.shape[0]
    if lastblock.downsample is not None:
        filters = filters // 2
    return filters


class ResnetBackbone(AbstractBackbone):
    def __init__(self, model: models.ResNet):
        super().__init__()
        self.model = model
        self.filters = [64] + [
            count_filters(layer)
            for layer in [
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
            ]
        ]

    def forward(self, x):
        # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
        m = self.model
        x = m.conv1(x)  # conv stride 2, effective 2
        x = m.bn1(x)
        x = m.relu(x)
        out_stride_2 = x
        x = m.maxpool(x)  # stride 2, effective 4
        x = m.layer1(x)  # stride 1, effective 4
        out_stride_4 = x
        x = m.layer2(x)  # stride 2, effective 8
        out_stride_8 = x
        x = m.layer3(x)  # stride 2, effective 16
        out_stride_16 = x
        x = m.layer4(x)  # stride 2, effective 32
        out_stride_32 = x
        return (
            out_stride_2,
            out_stride_4,
            out_stride_8,
            out_stride_16,
            out_stride_32,
        )
