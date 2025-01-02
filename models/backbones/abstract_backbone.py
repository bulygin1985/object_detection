from abc import abstractmethod
from typing import Tuple

import torch.nn as nn
import torchvision.transforms.functional as F
from torch import Tensor


class AbstractBackbone(nn.Module):
    """
    Backbone should implement forward method returning features at different strides.
    The class should also have filters member (list) with number of channels for each model
    ouput.
    """

    @abstractmethod
    def normalize(self, x: Tensor):
        """Normalize (preprocess) image tensor.
        Args:
            x (torch.Tensor): input float image scaled to [0, 1] range.
        returns image in a form that is ready to be processed by forward method.
        """
        pass

    @abstractmethod
    def forward(self, x: Tensor):
        """run feature extraction on prepared image x.
        returns features at strides 2, 4, 8, 16, 32
        """
        pass


imagenent_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
imagenent_std: Tuple[float, ...] = (0.229, 0.224, 0.225)


class ImageNetNormalizedBackbone(AbstractBackbone):
    """Backbone with normalization according to imagenet means and std."""

    def normalize(self, x: Tensor):
        return F.normalize(x, imagenent_mean, imagenent_std)


class ConcreteTorchvisionBackbone(ImageNetNormalizedBackbone):
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

    def forward(self, x: Tensor):
        strided_outputs = []
        prev_layer = 0
        for layer_no in self.stride_features_no:
            for layer in self.model.features[prev_layer:layer_no]:
                x = layer(x)
            prev_layer = layer_no
            strided_outputs.append(x)
        return strided_outputs
