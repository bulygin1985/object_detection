import torchvision.models as models

from .abstract_backbone import AbstractBackbone


class ResnetBackbone(AbstractBackbone):
    def __init__(self, model: models.ResNet):
        super().__init__()
        self.model = model
        self.filters = [64, 64, 128, 256, 512]

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
