import torch
import torchvision
import torchvision.models as models
from torchvision.models.mobilenetv2 import InvertedResidual

from .abstract_backbone import ConcreteTorchvisionBackbone


def layer_input_filters_and_stride(layer):
    """Get number of input filters (channels) and local stride for the input layer.
    Args:
        layer: model layer at some level (InvertedResidual, Conv2dNormActivation, Conv2d, etc).
    Returns:
        int: number of input filters and stride for the layer.
    """
    if isinstance(layer, torchvision.ops.Conv2dNormActivation):
        input_filters = layer[0].weight.shape[1]
        stride = layer[0].stride[0]
        return input_filters, stride
    elif isinstance(layer, torch.nn.Conv2d):
        return layer.weight.shape[1], layer.stride[0]
    elif isinstance(layer, InvertedResidual):
        input_filters = layer.conv[0][0].weight.shape[1]
        stride = 1
        for nestedlayer in layer.conv:
            _, nested_stride = layer_input_filters_and_stride(nestedlayer)
            stride = max(stride, nested_stride)
        return input_filters, stride
    else:
        return 0, 0


def get_stride_features_and_filters(model: models.MobileNetV2):
    """Get strided 'features' numbers (position in nn.Sequential) and number of input channels for them.
    Args:
        model (models.MobileNetV2): model
    Returns:
        List[int], List[int]: layer numbers (position) and number of input channels for strided layers.
    """
    # https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
    output_number = []
    nfilters = []
    for n, feature in enumerate(model.features):
        input_filters, stride = layer_input_filters_and_stride(feature)
        if stride == 2:
            output_number.append(n)
            nfilters.append(input_filters)
    # assume that the very last feature layer (channel expansion to 1280)
    # actually creates features for classification and is redundant
    output_number.append(len(model.features) - 1)
    nfilters.append(layer_input_filters_and_stride(model.features[-1])[0])
    return output_number[1:], nfilters[1:]  # ignore stride 1 data


def create_mobilenetv2_backbone(name: str, weights: str = None):
    assert name == "mobilenet_v2"
    model = models.get_model(name, weights=weights)
    return ConcreteTorchvisionBackbone(model, get_stride_features_and_filters)
