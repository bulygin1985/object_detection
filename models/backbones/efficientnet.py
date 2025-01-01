import torch
import torchvision
import torchvision.models as models
from torchvision.models.efficientnet import FusedMBConv, MBConv

from .abstract_backbone import ConcreteTorchvisionBackbone


def layer_props_en(layer):
    if isinstance(layer, torchvision.ops.Conv2dNormActivation):
        return layer[0].weight.shape[1], layer[0].stride[0]
    if isinstance(layer, torch.nn.Conv2d):
        return layer.weight.shape[1], layer.stride[0]
    is_sequential = isinstance(layer, torch.nn.Sequential)
    if is_sequential or isinstance(layer, MBConv) or isinstance(layer, FusedMBConv):
        blocks = layer if is_sequential else layer.block
        input_channels, stride = layer_props_en(blocks[0])
        for layer in blocks[1:]:
            _, layer_stride = layer_props_en(layer)
            stride = max(stride, layer_stride)
        return input_channels, stride
    else:
        return 0, 0


def get_stride_features_and_filters_en(model: models.EfficientNet):
    # https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py
    output_number = []
    nchannels = []
    for n, feature in enumerate(model.features):
        input_channels, stride = layer_props_en(feature)
        if stride == 2:
            output_number.append(n)
            nchannels.append(input_channels)
    # assume that the very last feature layer (channel expansion to 1280)
    # actually creates features for classification and is redundant
    output_number.append(len(model.features) - 1)
    nchannels.append(layer_props_en(model.features[-1])[0])
    return output_number[1:], nchannels[1:]  # ignore stride 1 data


def create_efficientnet_backbone(name, weights=None):
    assert name.startswith("efficientnet")
    model = models.get_model(name, weights=weights)
    return ConcreteTorchvisionBackbone(model, get_stride_features_and_filters_en)
