from collections import OrderedDict

import torch
import torch.nn as nn


class CenternetTTFLoss(nn.Module):
    def __init__(
        self,
        class_num,
        down_ratio,
        out_height,
        out_width,
        loss_dict={},
        ground_truth_bbox_encoding="absolute",
    ):
        super().__init__()

        cols = torch.arange(out_width)
        for _ in range(out_height - 1):
            cols = torch.vstack((cols, torch.arange(out_width)))

        rows = torch.arange(out_height)
        for _ in range(out_width - 1):
            rows = torch.hstack((rows, torch.arange(out_height)))
        rows = rows.view(out_width, out_height).T

        self._cols = cols
        self._rows = rows

        self._down_ratio = down_ratio

        self.loss_dict = loss_dict
        if self.loss_dict == None:
            self.loss_dict = {}
        assert (
            ground_truth_bbox_encoding == "absolute"
            or ground_truth_bbox_encoding == "relative"
        )
        self.ground_truth_bbox_as_offsets = ground_truth_bbox_encoding == "relative"

        self._class_num = class_num

        self.reg_loss = loss_dict.get("reg_loss", "l1")
        self.lambda_size = loss_dict.get("lambda_size", 0.1)
        self.lambda_cls = loss_dict.get("lambda_cls", 1.0)
        self.delta = 1e-5

        self._losses = OrderedDict({k: 0.0 for k in ["loss_cls", "loss_box", "loss"]})

    def get_box_coors(self, y_pred):
        """
        :param y_pred: detector output

        :return: output tensor T is H x W x 4, where W, H is width, heith of the model output
        each vector T [i, j, :] is box coordinates (x_min, y_min, x_max, y_max), which are true only at the GT box centers
            the predicted coor_pred = (x_min, y_min, x_max, y_max) is calculated using y_pred by formula:
            x_min = col * r - w_l
            y_min = row * r - h_t
            x_max = col * r + w_r
            y_max = row * r + h_b,
            where:
                row, col - GT row col of the object
                r is down_ratio, i.e. stride
                y_pred = (w_l, h_t, w_r, h_b) (w_l - width left, h_t - height top, ...)
        """

        if y_pred.get_device() != -1:
            self._cols = self._cols.to(y_pred.get_device())
            self._rows = self._rows.to(y_pred.get_device())

        x1 = self._down_ratio * self._cols - y_pred[..., 0]
        y1 = self._down_ratio * self._rows - y_pred[..., 1]
        x2 = self._down_ratio * self._cols + y_pred[..., 2]
        y2 = self._down_ratio * self._rows + y_pred[..., 3]

        res = torch.stack([x1, y1, x2, y2])
        res = res.permute((1, 2, 3, 0))

        return res

    def focal_loss(self, y_true, y_pred):
        """
        :param y_true: encoded GT
        :param y_pred: detector output
        :return: float number which corrensponds to object class prediction error
                 below is sum by row, col
                 -sum[ log(y_pred) * (1 - y_pred)^2 ] / N if y_true=1
                 -sum[ log(1 - y_pred) *  y_pred^2 * (1 - y_true)^4 *] / N otherwise
                 where N is number of objects
        """
        pos_inds = y_true.eq(1.0).float()

        neg_inds = 1.0 - pos_inds
        neg_weights = torch.pow(1.0 - y_true, 4.0)

        y_pred = torch.clamp(y_pred, self.delta, 1.0 - self.delta)
        pos_loss = torch.log(y_pred) * torch.pow(1.0 - y_pred, 2.0) * pos_inds
        neg_loss = (
            torch.log(1.0 - y_pred) * torch.pow(y_pred, 2.0) * neg_weights * neg_inds
        )

        _sum = pos_inds.sum()
        num_pos = torch.maximum(_sum, torch.ones_like(_sum))
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        loss = -(pos_loss + neg_loss) / num_pos
        return loss

    def reg_l1_loss(self, y_true, y_pred):
        """
        :param y_true: encoded GT
        :param y_pred: detector output
        :return: -sum( |y_true-y_pred| ) , where sum is only for GT box centers
        """

        mask = torch.gt(y_true, 0.0).float()
        num_pos = mask.sum()
        if torch.eq(num_pos, 0):
            return 0.0

        res = (
            y_pred if self.ground_truth_bbox_as_offsets else self.get_box_coors(y_pred)
        )
        loss = torch.abs(res - y_true) * mask

        loss = loss.sum() / num_pos
        return loss

    def reg_iou_loss(self, y_true, y_pred):
        y_true = torch.reshape(y_true, (-1, 4))
        _y_true = torch.sum(y_true, dim=1)

        mask = torch.gt(_y_true, 0.0).float()
        num_pos = torch.sum(mask)

        if num_pos == 0:
            return 0.0

        y_pred = self.get_box_coors(y_pred)
        y_pred = torch.reshape(y_pred, (-1, 4))

        x1g, y1g, x2g, y2g = y_true[:, 0], y_true[:, 1], y_true[:, 2], y_true[:, 3]
        x1, y1, x2, y2 = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]

        xA = torch.maximum(x1g, x1)
        yA = torch.maximum(y1g, y1)
        xB = torch.minimum(x2g, x2)
        yB = torch.minimum(y2g, y2)

        _zero = torch.zeros_like(xA)
        interArea = torch.maximum(_zero, (xB - xA + 1)) * torch.maximum(
            _zero, yB - yA + 1
        )

        boxAArea = torch.maximum(_zero, (x2g - x1g + 1)) * torch.maximum(
            _zero, (y2g - y1g + 1)
        )
        boxBArea = torch.maximum(_zero, (x2 - x1 + 1)) * torch.maximum(
            _zero, (y2 - y1 + 1)
        )

        iou = interArea / (boxAArea + boxBArea - interArea + self.delta)

        loss = -torch.log((iou * mask).sum() / num_pos)

        return loss

    def forward(self, y_true, y_pred):
        """
        :param y_true: encoded GT
        :param y_pred: detector output
        :return: focal_loss + lambda*reg_l1_loss
        """
        y_true = y_true.float()
        y_pred = y_pred.float()
        y_pred = y_pred.permute(0, 2, 3, 1)

        c = self._class_num

        hm_loss = self.focal_loss(y_true[..., :c], y_pred[..., :c])

        if self.reg_loss == "l1":
            coor_loss = self.reg_l1_loss(y_true[..., c:], y_pred[..., c:])
        elif self.reg_loss == "iou":
            coor_loss = self.reg_iou_loss(y_true[..., c:], y_pred[..., c:])
        else:
            raise Exception("loss_params['reg_loss'] must be from the list : [l1, iou]")

        self._losses["loss_cls"] = hm_loss
        self._losses["loss_box"] = coor_loss
        self._losses["loss"] = self.lambda_cls * hm_loss + self.lambda_size * coor_loss

        return self._losses
