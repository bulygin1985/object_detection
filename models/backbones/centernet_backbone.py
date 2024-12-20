from collections import OrderedDict

import numpy as np
import torch.nn as nn


class Backbone(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.block_num = 1
        self.alpha = alpha
        self.filters = np.array(
            [
                64 * self.alpha,
                128 * self.alpha,
                256 * self.alpha,
                512 * self.alpha,
                512 * self.alpha,
            ]
        ).astype("int")
        s = self.filters
        self.layer1 = self.conv_bn_relu(3, s[0], False)
        self.layer2 = self.conv_bn_relu(s[0], s[0], True)  # stride 2
        self.layer3 = self.conv_bn_relu(s[0], s[1], False)
        self.layer4 = self.conv_bn_relu(s[1], s[1], True)  # stride 4
        self.layer5 = self.conv_bn_relu(s[1], s[2], False)
        self.layer6 = self.conv_bn_relu(s[2], s[2], False)
        self.layer7 = self.conv_bn_relu(s[2], s[2], True)  # stride 8
        self.layer8 = self.conv_bn_relu(s[2], s[3], False)
        self.layer9 = self.conv_bn_relu(s[3], s[3], False)
        self.layer10 = self.conv_bn_relu(s[3], s[3], True)  # stride 16
        self.layer11 = self.conv_bn_relu(s[4], s[4], False)
        self.layer12 = self.conv_bn_relu(s[4], s[4], False)
        self.layer13 = self.conv_bn_relu(s[4], s[4], True)  # stride 32

    def conv_bn_relu(self, input_num, output_num, max_pool=False, kernel_size=3):
        block = OrderedDict()
        block["conv_" + str(self.block_num)] = nn.Conv2d(
            input_num, output_num, kernel_size=kernel_size, stride=1, padding=1
        )
        block["bn_" + str(self.block_num)] = nn.BatchNorm2d(
            output_num, eps=1e-3, momentum=0.01
        )
        block["relu_" + str(self.block_num)] = nn.ReLU()
        if max_pool:
            block["pool_" + str(self.block_num)] = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block_num += 1
        return nn.Sequential(block)

    def forward(self, x):
        out = self.layer1(x)
        out_stride_2 = self.layer2(out)
        out = self.layer3(out_stride_2)
        out_stride_4 = self.layer4(out)
        out = self.layer5(out_stride_4)
        out = self.layer6(out)
        out_stride_8 = self.layer7(out)
        out = self.layer8(out_stride_8)
        out = self.layer9(out)
        out_stride_16 = self.layer10(out)
        out = self.layer11(out_stride_16)
        out = self.layer12(out)
        out_stride_32 = self.layer13(out)
        return out_stride_2, out_stride_4, out_stride_8, out_stride_16, out_stride_32
