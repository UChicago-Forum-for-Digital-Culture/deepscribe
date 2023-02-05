from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import VisionDataset
from PIL import Image
from typing import Optional, Callable
import torch
import json


class CuneiformLocalizationDataset(VisionDataset):
    """
    Object detection dataset for labeled cuneiform tablets.

    """

    def __init__(
        self,
        labels: str,
        root: str,
        localization_only=False,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.localization_only = localization_only

        # open labels file
        with open(labels) as inf:
            self.data = json.load(inf)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]
        img = Image.open(f"{self.root}/{entry['file_name']}")
        bboxes, labels = zip(
            *[
                (
                    annotation["bbox"],
                    annotation["category_id"] if not self.box_only else 0,
                )
                for annotation in entry["annotations"]
            ]
        )

        boxes = torch.tensor(bboxes)
        labels = torch.tensor(labels, dtype=torch.int64)
        # apply transforms if present
        if self.transforms:
            img, boxes, labels = self.transforms(img, boxes, labels)

        targets = {"boxes": boxes, "labels": labels}

        return img, targets


def collate_retinanet(batch_input):
    images, targets = zip(
        *[
            (
                img,
                {key: val for key, val in lab.items()},
            )
            for img, lab in batch_input
        ]
    )
    return list(images), list(targets)
