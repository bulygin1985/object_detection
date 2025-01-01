from abc import abstractmethod

import torch.nn as nn


class AbstractBackbone(nn.Module):
    """
    Backbone should implement forward method returning features at different strides.
    The class should also have filters member (list) with number of channels for each model
    ouput.
    """

    @abstractmethod
    def forward(self, x):
        """run feature extraction on prepared image x.
        returns features at strides 2, 4, 8, 16, 32
        """
        pass


class ConcreteTorchvisionBackbone(AbstractBackbone):
    """Default backbone implementation for torchvision model.
    Fits for models that contain feature extraction layers in features member(nn.Sequential).
    """

    def __init__(self, model: nn.Module, model_stride_features_and_filters_func):
        """Initialize backbone.
        Args:
            model (nn.Module): model object, see class docstring for constraints.
            model_stride_features_and_filters_func: function returning strided layer
                numbers and number of channels in them. Function signature:
                def get_stide_layers_and_channels(model) -> Tuple(List[int], List[int])
        """
        super().__init__()
        self.model = model
        layers_no, filters_count = model_stride_features_and_filters_func(model)
        self.stride_features_no = layers_no
        self.filters = filters_count

    def forward(self, x):
        strided_outputs = []
        prev_layer = 0
        for layer_no in self.stride_features_no:
            for layer in self.model.features[prev_layer:layer_no]:
                x = layer(x)
            prev_layer = layer_no
            strided_outputs.append(x)
        return strided_outputs
