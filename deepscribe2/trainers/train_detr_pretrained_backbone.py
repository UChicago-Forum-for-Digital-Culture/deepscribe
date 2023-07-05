from pathlib import Path

import pytorch_lightning as pl
import torch

import wandb
from deepscribe2 import transforms as T
from deepscribe2.datasets import PFADetectionDataModule
from deepscribe2.models.detection import DETRLightningModule
from deepscribe2.utils import load_ckpt_from_wandb

DATA_BASE = "/local/ecw/DeepScribe_Data_2023-02-04-selected"
WANDB_PROJECT = "deepscribe-torchvision"
MONITOR_ATTRIBUTE = "map_50"
LOCALIZATION_ONLY = False

xforms = T.Compose(
    [
        # T.RandomHorizontalFlip(),
        T.RandomShortestSize(
            [500, 640, 672, 704, 736, 768, 800], 1333
        ),  # taken directly from detectron2 config.
        # T.RandomIoUCrop(),
        # T.RandomZoomOut(),
        # T.RandomPhotometricDistort(),
    ]
)

pfa_data_module = PFADetectionDataModule(
    DATA_BASE,
    autocrop=True,
    batch_size=1,
    train_xforms=xforms,
    localization_only=LOCALIZATION_ONLY,
    collate_fn="detr",
    start_from_one=True,
)

model = DETRLightningModule(num_classes=pfa_data_module.num_labels)

# load artifact!!

ckpt = load_ckpt_from_wandb("model-01ax80np:v13")

model.init_backbone_from_retinanet_state(ckpt["state_dict"])


logger = pl.loggers.WandbLogger(project=WANDB_PROJECT, log_model="all")

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor=MONITOR_ATTRIBUTE, mode="max"
)
# earlystop_callback = pl.callbacks.EarlyStopping(
#     monitor=MONITOR_ATTRIBUTE, mode="max", patience=5
# )

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    logger=logger,
    max_epochs=250,
    callbacks=[checkpoint_callback],
    gradient_clip_val=0.1,
)
trainer.fit(model, datamodule=pfa_data_module)
