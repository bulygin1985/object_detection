import numpy as np


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma_x=1, sigma_y=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-x * x / (2 * sigma_x**2) - y * y / (2 * sigma_y**2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius_x, radius_y):
    diameter_x = 2 * radius_x + 1
    diameter_y = 2 * radius_y + 1
    gaussian = gaussian2D(
        (diameter_y, diameter_x), sigma_y=diameter_y / 6, sigma_x=diameter_x / 6
    )

    x, y = int(np.round(center[0])), int(np.round(center[1]))

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius_x), min(width - x, radius_x + 1)
    top, bottom = min(y, radius_y), min(height - y, radius_y + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius_y - top : radius_y + bottom, radius_x - left : radius_x + right
    ]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
    return heatmap


class CenternetEncoder:
    def __init__(
        self,
        img_height=320,
        img_width=320,
        down_ratio=4,
        n_classes=20,
        coords_relative: bool = False,
        coords_scaled: bool = False,
    ):
        """
        Args:
            coords_relative: when True encode bbox edges(l,t,b,r) relative to the center of bb center cell.
            coords_scaled: when True encode bbox edges(l,t,b,r) divided by image dimension
        """
        self._img_height = img_height
        self._img_width = img_width
        self._out_height = self._img_height // down_ratio
        self._out_width = self._img_width // down_ratio
        self._n_classes = n_classes
        self._down_ratio = down_ratio
        self._coords_relative = coords_relative
        self._coords_scaled = coords_scaled
        print("down_ratio = {}".format(self._down_ratio))

    def __call__(self, bboxes, labels):
        hm = np.zeros(
            (self._out_height, self._out_width, self._n_classes), dtype=np.float32
        )
        coors = np.zeros((self._out_height, self._out_width, 4), dtype=np.float32)
        for cls_id, bbox in zip(labels.data.numpy(), bboxes.data.numpy()):
            # print(bbox)
            box_s = bbox / self._down_ratio
            h, w = box_s[3] - box_s[1], box_s[2] - box_s[0]
            rad_w_class = int(np.round(gaussian_radius([h, w])))
            rad_h_class = rad_w_class
            if h > 0 and w > 0:
                center = np.array(
                    [(box_s[0] + box_s[2]) / 2, (box_s[1] + box_s[3]) / 2],
                    dtype=np.float32,
                )
                if self._coords_relative:
                    center = np.floor(center)
                else:
                    center = np.round(center)
                center = np.clip(
                    center, [0, 0], [self._out_width - 1, self._out_height - 1]
                )
                center_int = center.astype(np.int32)
                draw_gaussian(hm[..., cls_id - 1], center_int, rad_w_class, rad_h_class)
                if self._coords_scaled or self._coords_relative:
                    bbox = bbox.copy()
                    if self._coords_relative:
                        x_center_cell = (center[0] + 0.5) * self._down_ratio
                        y_center_cell = (center[1] + 0.5) * self._down_ratio
                        bbox[0] = x_center_cell - bbox[0]
                        bbox[2] = bbox[2] - x_center_cell
                        bbox[1] = y_center_cell - bbox[1]
                        bbox[3] = bbox[3] - y_center_cell
                    if self._coords_scaled:
                        bbox[0] /= self._img_width
                        bbox[2] /= self._img_width
                        bbox[1] /= self._img_height
                        bbox[3] /= self._img_height
                coors[center_int[1], center_int[0]] = bbox
        return np.concatenate((hm, coors), axis=-1)
