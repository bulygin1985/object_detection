import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CenternetPostprocess(nn.Module):
    detection_num = 100
    """
    Layer with input of CenterNet output.The layer output is tensor [batch_size, k, 6]
    where 'k 'is max object number, '6' : (class_id, score, x_min, y_min, x_max, y_max)
    """
    def __init__(self, n_classes=80, width=320, height=320, down_ratio=4):
        super().__init__()
        self._n_classes = n_classes
        self._down_ratio = down_ratio
        self._width = width
        self._height = height
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def forward(self, y_pred):
        hm = y_pred[:, :self._n_classes, :, :]
        coors = y_pred[:, self._n_classes: self._n_classes+4, :, :]
        return self._ctdet_decode(hm, coors, output_stride=self._down_ratio, k=self.detection_num)
    
    def _nms(self, hm, kernel=3):
        hmax = F.max_pool2d(hm, kernel, stride=1, padding=kernel//2)
        keep = torch.eq(hmax, hm).type(torch.float32).to(self.device)
        hm = hm.type(torch.float32)
        return hm * keep
    
    def _ctdet_decode(self, hm, coors, output_stride, k):
        """
        :param hm: class heatmaps [batch_size, rows, cols, class_number]
        :param coors : width, height heatmaps [batch_size, rows, cols, 2] - 1st for width, second for height
        :param output_stride: ratio between detector input and output image sizes
        :param k: maximum number od detections on single image
        :return: set of [class_id, score, x_min, y_min, x_max, y_max]
        """
        hm = self._nms(hm)
        hm = hm.permute(0, 2, 3, 1)
        coors = coors.permute(0, 2, 3, 1)
        hm_shape = hm.shape
        coors_shape = coors.shape
        batch, width, cat = hm_shape[0], hm_shape[2], hm_shape[3]
        hm_flat = torch.reshape(hm, (batch, -1))
        coors_flat = torch.reshape(coors, (coors_shape[0], -1, coors_shape[-1]))
        
        def _process_sample(args):
            _hm, _coors = args
            _scores, _inds = torch.topk(_hm, k=k, dim=1, sorted=True)
            _classes = (_inds % cat).type(torch.float32) + 1.0
            _inds = (_inds / cat).type(torch.int64)
            _xs = (_inds % width).type(torch.float32)
            _ys = (_inds / width).type(torch.int32).type(torch.float32)
            _x1 = (output_stride * _xs - _coors[..., 0].gather(dim=1, index=_inds).type(torch.float32)) / self._width
            _y1 = (output_stride * _ys - _coors[..., 1].gather(dim=1, index=_inds).type(torch.float32)) / self._height
            _x2 = (output_stride * _xs + _coors[..., 2].gather(dim=1, index=_inds).type(torch.float32)) / self._width
            _y2 = (output_stride * _ys + _coors[..., 3].gather(dim=1, index=_inds).type(torch.float32)) / self._height
            # _classes : integer class number
            # _x1, _y1, _x2, _y2 : x_min, y_min, x_max, y_max
            # _ys, _xs : integer row, col of the gaussian center
            _detection = torch.stack([_classes, _scores, _x1, _y1, _x2, _y2, _ys, _xs], -1)
            return _detection
        detections = _process_sample([hm_flat, coors_flat])
        return detections