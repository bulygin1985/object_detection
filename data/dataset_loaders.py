from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2


class MSCOCODatasetLoader:
    def __init__(self, img_folder: str, ann_file: str):
        self.img_folder = img_folder
        self.ann_file = ann_file

    def get_dataset(self):
        raw_ds = CocoDetection(
            root=self.img_folder,
            annFile=self.ann_file,
        )
        return wrap_dataset_for_transforms_v2(raw_ds)
