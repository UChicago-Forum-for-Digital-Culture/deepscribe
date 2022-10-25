from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import json


class DeepScribeDataset(Dataset):
    def __init__(self, data_files, img_folder, box_only=False):
        super().__init__()
        # detectron2-formatted dataset
        if isinstance(data_files, str):
            data_files = [data_files]

        self.data = []
        for dfile in data_files:
            with open(dfile, "r") as inf:
                self.data.extend(json.load(inf))

        self.img_folder = img_folder

        self.transforms = transforms.Compose([transforms.PILToTensor()])
        self.box_only = box_only

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]
        img = Image.open(f"{self.img_folder}/{entry['file_name']}")

        bboxes, labels = zip(
            *[
                (
                    annotation["bbox"],
                    annotation["category_id"] if not self.box_only else 0,
                )
                for annotation in entry["annotations"]
            ]
        )
        # Actual normalization is handled by the retinanet module
        return (self.transforms(img) / 255.0), {
            "boxes": torch.tensor(bboxes),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }


def collate_retinanet(batch_input):

    images, labels = zip(
        *[
            (
                img,
                {key: val for key, val in lab.items()},
            )
            for img, lab in batch_input
        ]
    )
    return list(images), list(labels)
