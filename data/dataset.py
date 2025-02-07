from collections import namedtuple

import torch
from torch.utils import data

Empty_object = namedtuple("Empty_object", [])
Empty_object.get = lambda *args: Empty_object()


class Dataset(data.Dataset):
    def __init__(self, dataset, transformation, encoder=None):
        self._dataset = dataset
        self._transformation = transformation
        self._encoder = encoder

    def __getitem__(self, index):
        img, lbl = self._dataset[index]
        img_, bboxes_, labels_ = self._transformation(
            img, lbl.get("boxes", []), lbl.get("labels", [])
        )

        if self._encoder is None:
            return img_, Empty_object()

        lbl_encoded = self._encoder(bboxes_, labels_)
        return img_, torch.from_numpy(lbl_encoded)

    def __len__(self):
        return len(self._dataset)
