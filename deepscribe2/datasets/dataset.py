import json
from typing import Callable, Optional, Dict, Any

import pandas as pd
import torch
from torchvision.datasets import VisionDataset
from torchvision.io import read_image

# from torchvision.datapoints import BoundingBox, BoundingBoxFormat


# # converts a dataset entry to the labels required by transforms v2.
# def entry_to_datapoints(
#     dataset_entry: Dict[str, Any], increase_label_by: int = 0
# ) -> Dict[str, Any]:
#     boxes = [
#         BoundingBox(
#             anno["bbox"],
#             format=BoundingBoxFormat.XYXY,
#             spatial_size=(dataset_entry["height"], dataset_entry["width"]),
#         )
#         for anno in dataset_entry["annotations"]
#     ]

#     labels = [
#         anno["category_id"] + increase_label_by for anno in dataset_entry["annotations"]
#     ]
#     return {"boxes": boxes, "labels": labels}


class CuneiformLocalizationDataset(VisionDataset):
    """
    Object detection dataset for labeled cuneiform tablets.

    """

    def __init__(
        self,
        labels: str,
        img_root: str,
        classes_info: str,
        localization_only=False,
        transforms: Optional[Callable] = None,
        start_from_one: bool = False,
    ) -> None:
        super().__init__(img_root, transforms, transform=None, target_transform=None)

        self.localization_only = localization_only
        self.start_from_one = start_from_one

        # load classes info
        # pandas will interpret the sign "NA" as NaN
        self.sign_data = pd.read_csv(
            classes_info, na_filter=False, names=["sign", "category_id"]
        )

        self.class_labels = self.sign_data["sign"].tolist()
        self.num_labels = 1 if self.localization_only else len(self.sign_data)
        if self.start_from_one:
            self.num_labels += 1

        # open labels file
        with open(labels) as inf:
            self.data = json.load(inf)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]

        # rescaling to between 0 and 1
        img = read_image(f"{self.root}/{entry['file_name']}") / 255.0

        # UPDATING TO MATCH LATEST RETINANET DOCS.
        # BACKGROUND CLASS IS ZERO!!!!
        # man that was annoying.

        # targets = entry_to_datapoints(
        #     entry, increase_label_by=1 if self.start_from_one else 0
        # )

        bboxes, labels = zip(
            *[
                (
                    annotation["bbox"],
                    annotation["category_id"] + int(self.start_from_one)
                    if not self.localization_only
                    else int(self.start_from_one),
                )
                for annotation in entry["annotations"]
            ]
        )

        targets = {
            "boxes": torch.tensor(bboxes).long(),
            "labels": torch.tensor(labels).long(),
        }
        # apply transforms if present
        if self.transforms:
            img, targets = self.transforms(img, targets)

        # return img size for DETR usage

        _, height, width = img.size()

        return img, targets, (height, width)


def collate_retinanet(batch_input):
    images, targets = zip(
        *[
            (
                img,
                {key: val for key, val in lab.items()},
            )
            for img, lab, _ in batch_input
        ]
    )
    return list(images), list(targets)
