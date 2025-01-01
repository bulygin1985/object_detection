import torch
import torchvision
import torchvision.models as models
from torchvision.models.mobilenetv2 import InvertedResidual, MobileNetV2 

from .abstract_backbone import AbstractBackbone


def layer_props(layer):
    if isinstance(layer, torchvision.ops.Conv2dNormActivation):
        # print('Conv2dNormActivation', layer[0].weight.shape, layer[0].stride)
        input_filters = layer[0].weight.shape[1]
        stride = layer[0].stride[0]
        return input_filters, stride
    elif isinstance(layer, torch.nn.Conv2d):
        # print('Conv2d', layer.weight.shape, layer.stride)
        return layer.weight.shape[1], layer.stride[0]
    elif isinstance(layer, InvertedResidual):
        # expect
        input_filters = layer.conv[0][0].weight.shape[1]
        stride = 1
        for nestedlayer in layer.conv:
            _, nested_stride = layer_props(nestedlayer)
            stride = max(stride, nested_stride)
        return input_filters, stride
    else:
        return 0, 0


def get_stride_features_and_filters(model):
    output_number = []
    nchannels = []
    for n, feature in enumerate(model.features):
        input_filters, stride = layer_props(feature)
        if stride == 2:
            output_number.append(n)
            nchannels.append(input_filters)
        # print('----')
    # assume that the very last feature layer (channel expansion to 1280)
    # actually creates features for classification and is redundant
    output_number.append(len(model.features) - 1)
    nchannels.append(layer_props(model.features[-1])[0])
    return output_number[1:], nchannels[1:]  # ignore stride 1 data


class MobileNetV2Backbone(AbstractBackbone):
    def __init__(self, model: models.MobileNetV2):
        super().__init__()
        self.model = model
        layers_no, filters_count = get_stride_features_and_filters(model)
        self.stride_features_no = layers_no
        self.filters = filters_count

    def forward(self, x):
        # https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
        strided_outputs = []
        prev_layer = 0
        for layer_no in self.stride_features_no:
            for layer in self.model.features[prev_layer:layer_no]:
                x = layer(x)
            prev_layer = layer_no
            strided_outputs.append(x)
        return strided_outputs


def create_mobilenetv2_backbone(name, weights=None):
    assert name == "mobilenet_v2"
    model = models.get_model(name, weights=weights)
    return MobileNetV2Backbone(model)
