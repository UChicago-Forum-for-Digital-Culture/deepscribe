import json
from abc import ABC
from copy import deepcopy
from typing import Dict, List, Tuple, Union

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image


# mostly for debugging. pulls data directly from annotations file.
class DirectHotspotDataset(Dataset):
    def __init__(self, dset_file, image_folder, transforms):
        super().__init__()

        with open(dset_file) as inf:
            dset = json.load(inf)

        self.hotspots = []

        for entry in dset:
            for anno in entry["annotations"]:
                anno["file_name"] = entry["file_name"]
                self.hotspots.append(anno)

        self.image_folder = image_folder
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.hotspots)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        entry = self.hotspots[idx]

        img = Image.open(self.image_folder + "/" + entry["file_name"])

        cutout = img.crop(box=entry["bbox"])

        return self.transforms(cutout), entry["category_id"]
