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
