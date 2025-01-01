from typing import List, Tuple

import torch
import torchvision
import torchvision.models as models
from torchvision.models.efficientnet import FusedMBConv, MBConv

from .abstract_backbone import ConcreteTorchvisionBackbone


def layer_input_filters_and_stride(layer) -> Tuple[int, int]:
    """Get number of input filters (channels) and local stride for the input layer.
    Args:
        layer: model layer at some level (nn.Sequential, MBConv, FusedMBConv, Conv2dNormActivation, Conv2d, etc).
    Returns:
        int: number of input filters and stride for the layer.
    """
    if isinstance(layer, torchvision.ops.Conv2dNormActivation):
        return layer[0].weight.shape[1], layer[0].stride[0]
    if isinstance(layer, torch.nn.Conv2d):
        return layer.weight.shape[1], layer.stride[0]
    is_sequential = isinstance(layer, torch.nn.Sequential)
    if is_sequential or isinstance(layer, MBConv) or isinstance(layer, FusedMBConv):
        blocks = layer if is_sequential else layer.block
        input_filters, stride = layer_input_filters_and_stride(blocks[0])
        for layer in blocks[1:]:
            _, layer_stride = layer_input_filters_and_stride(layer)
            stride = max(stride, layer_stride)
        return input_filters, stride
    else:
        return 0, 0


def get_stride_features_and_filters(
    model: models.EfficientNet,
) -> Tuple[List[int], List[int]]:
    """Get strided 'features' numbers (position in nn.Sequential) and number of input channels for them.
    Args:
        model (models.EfficientNet): model
    Returns:
        List[int], List[int]: layer numbers (position) and number of input channels for strided layers.
    """
    # https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py
    output_number = []
    nfilters = []
    for n, feature in enumerate(model.features):
        input_channels, stride = layer_input_filters_and_stride(feature)
        if stride == 2:
            output_number.append(n)
            nfilters.append(input_channels)
    # assume that the very last feature layer (channel expansion to 1280)
    # actually creates features for classification and is redundant
    output_number.append(len(model.features) - 1)
    nfilters.append(layer_input_filters_and_stride(model.features[-1])[0])
    return output_number[1:], nfilters[1:]  # ignore stride 1 data


def create_efficientnet_backbone(name: str, weights: str = None):
    assert name.startswith("efficientnet")
    model = models.get_model(name, weights=weights)
    return ConcreteTorchvisionBackbone(model, get_stride_features_and_filters)
