import abc

import torch.nn as nn


class AbstractBackbone(nn.Module):

    @abc.abstractmethod
    def forward(self, x):
        """run feature extraction on prepared image x.
        returns features at strides 2, 4, 8, 16, 32
        """
        pass
