import torchvision.models as models

from .abstract_backbone import AbstractBackbone


def layer_out_width(resnet_layer):
    lastblock = resnet_layer[-1]
    if isinstance(lastblock, models.resnet.BasicBlock):
        width = lastblock.conv2.weight.shape[0]
    elif isinstance(lastblock, models.resnet.Bottleneck):
        width = lastblock.conv3.weight.shape[0]
    if lastblock.downsample is not None:
        print("downsample")
        width = width // 2
    return width


class ResnetBackbone(AbstractBackbone):
    def __init__(self, model: models.ResNet):
        super().__init__()
        self.model = model
        self.filters = [64] + [
            layer_out_width(layer)
            for layer in [model.layer1, model.layer2, model.layer3, model.layer4]
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
        return out_stride_2, out_stride_4, out_stride_8, out_stride_16, out_stride_32


def create_resnet_backbone(name, weights=None):
    assert name.startswith("resnet")
    model = models.get_model(name, weights=weights)
    return ResnetBackbone(model)
