from torch.utils import data


class ImageDatasetWithLabels(data.Dataset):
    def __init__(self, dataset, transformation):
        self._dataset = dataset
        self._transformation = transformation

    def __getitem__(self, index):
        img, _ = self._dataset[index]
        return self._transformation(img)

    def __len__(self):
        return len(self._dataset)
