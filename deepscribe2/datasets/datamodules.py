import json
import os
from typing import Callable, Optional

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.io import write_jpeg
from tqdm import tqdm
import pandas as pd

from deepscribe2.preprocessing.crop_images import crop_to_box_boundaries
from deepscribe2.preprocessing.split_tablet import split_by_tablet
from deepscribe2.utils import get_boxes
from deepscribe2.datasets.dataset import CuneiformLocalizationDataset, collate_retinanet

generator = torch.Generator().manual_seed(42)

HOTSPOT_BASE_FILE = "imagesWithHotspots.json"
HOTSPOTS_CROPPED_BASE_FILE = "imagesWithHotspots_cropped.json"
IMAGES_BASE_DIR = "images"
IMAGES_CROPPED_DIR = "cropped_images"
CATEGORIES_BASE_FILE = "categories.txt"


class PFADetectionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        base_dir: str,
        autocrop: bool = True,
        autocrop_margin: int = 50,
        batch_size: int = 5,
        splits=(0.8, 0.1, 0.1),  # train, val, test
        train_xforms: Optional[Callable] = None,
        localization_only: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        self._check_base_dir()
        self._load_classes()

    def _check_base_dir(self):
        # check if required directories exist
        if not os.path.isdir(f"{self.hparams.base_dir}/{IMAGES_BASE_DIR}"):
            raise ValueError(
                f"base folder {self.hparams.base_dir} does not contain the expected images folder: {IMAGES_BASE_DIR}"
            )
        if not os.path.exists(f"{self.hparams.base_dir}/{HOTSPOT_BASE_FILE}"):
            raise ValueError(
                f"base folder {self.hparams.base_dir} does not contain the expected data file: {HOTSPOT_BASE_FILE}"
            )
        if not os.path.exists(f"{self.hparams.base_dir}/{CATEGORIES_BASE_FILE}"):
            raise ValueError(
                f"base folder {self.hparams.base_dir} does not contain the expected labels file: {CATEGORIES_BASE_FILE}"
            )

    def _load_classes(self):
        # load classes info
        # pandas will interpret the sign "NA" as NaN
        sign_data = pd.read_csv(
            self.categories_file, na_filter=False, names=["sign", "category_id"]
        )
        self.class_labels = sign_data["sign"].tolist()
        self.num_labels = 1 if self.hparams.localization_only else len(self.sign_data)

    @property
    def image_dir(self):
        suffix = IMAGES_CROPPED_DIR if self.hparams.autocrop else IMAGES_BASE_DIR
        return f"{self.hparams.base_dir}/{suffix}"

    @property
    def hotspot_file(self):
        suffix = (
            HOTSPOTS_CROPPED_BASE_FILE if self.hparams.autocrop else HOTSPOT_BASE_FILE
        )
        return f"{self.hparams.base_dir}/{suffix}"

    @property
    def categories_file(self):
        return f"{self.hparams.base_dir}/{CATEGORIES_BASE_FILE}"

    @property
    def train_partition(self):
        suffix = (
            "data_train_cropped.json" if self.hparams.autocrop else "data_train.json"
        )
        return f"{self.hparams.base_dir}/{suffix}"

    @property
    def val_partition(self):
        suffix = "data_val_cropped.json" if self.hparams.autocrop else "data_val.json"
        return f"{self.hparams.base_dir}/{suffix}"

    @property
    def test_partition(self):
        suffix = "data_test_cropped.json" if self.hparams.autocrop else "data_test.json"
        return f"{self.hparams.base_dir}/{suffix}"

    def _handle_autocrop(self):
        # if autocrop, run autocropping!
        if self.hparams.autocrop:
            if not os.path.isdir(f"{self.hparams.base_dir}/{IMAGES_CROPPED_DIR}"):
                print("Autocropping images! ")
                # TODO: error handling + cleanup!!
                os.mkdir(f"{self.hparams.base_dir}/{IMAGES_CROPPED_DIR}")

                cropped_entries = []

                with open(
                    f"{self.hparams.base_dir}/{HOTSPOT_BASE_FILE}", "r"
                ) as infile:
                    raw_dset = json.load(infile)

                for entry in tqdm(raw_dset):
                    new_entry, cropped_image = crop_to_box_boundaries(
                        entry,
                        f"{self.hparams.base_dir}/{IMAGES_BASE_DIR}",
                        margin=self.hparams.autocrop_margin,
                    )
                    outpath = f"{self.hparams.base_dir}/{IMAGES_CROPPED_DIR}/{new_entry['file_name']}"
                    cropped_entries.append(new_entry)
                    write_jpeg(cropped_image, outpath)

                with open(
                    {self.hparams.base_dir} / {HOTSPOTS_CROPPED_BASE_FILE}, "w"
                ) as outf:
                    json.dump(cropped_entries, outf)
            else:
                print("Autocrop directory found. Skipping.")

    def _handle_split(self):
        with open(self.hotspot_file) as inf:
            all_data = json.load(inf)

        if len(self.hparams.splits) != 3:
            raise ValueError(
                f"only 3 splits must be provided, {self.hparams.splits} were provided.."
            )

        if (
            os.path.exists(self.train_partition)
            or os.path.exists(self.val_partition)
            or os.path.exists(self.val_partition)
        ):
            print("Split files already exist, skipping.")
        else:
            print("running tablet split on data.")
            train_fold, val_fold, test_fold = split_by_tablet(
                all_data, self.hparams.splits
            )

            with open(self.train_partition, "w") as outf:
                json.dump(train_fold, outf)

            with open(self.val_partition, "w") as outf:
                json.dump(val_fold, outf)

            with open(self.test_partition, "w") as outf:
                json.dump(test_fold, outf)

    def prepare_data(self) -> None:
        # if autocrop, run cropping!
        self._handle_autocrop()
        self._handle_split()

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = CuneiformLocalizationDataset(
                self.train_partition,
                self.image_dir,
                self.categories_file,
                transforms=self.hparams.train_xforms,
                localization_only=self.hparams.localization_only,
            )

            self.val_dataset = CuneiformLocalizationDataset(
                self.val_partition,
                self.image_dir,
                self.categories_file,
                transforms=None,
                localization_only=self.hparams.localization_only,
            )
        if stage == "test":
            self.test_dataset = CuneiformLocalizationDataset(
                self.test_partition,
                self.image_dir,
                self.categories_file,
                transforms=None,
                localization_only=self.hparams.localization_only,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=5,
            shuffle=True,
            collate_fn=collate_retinanet,
            num_workers=12,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=5,
            shuffle=False,
            collate_fn=collate_retinanet,
            num_workers=12,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=5,
            shuffle=False,
            collate_fn=collate_retinanet,
            num_workers=12,
        )
