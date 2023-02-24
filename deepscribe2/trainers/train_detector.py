from pathlib import Path

import pytorch_lightning as pl

import wandb
from deepscribe2 import transforms as T
from deepscribe2.datasets import PFADetectionDataModule
from deepscribe2.models import RetinaNet

DATA_BASE = "/local/ecw/DeepScribe_Data_2023-02-04-selected"
WANDB_PROJECT = "deepscribe-torchvision"
MONITOR_ATTRIBUTE = "map_50"
LOCALIZATION_ONLY = False

xforms = T.Compose(
    [
        T.RandomHorizontalFlip(),
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
    batch_size=2,
    train_xforms=xforms,
    localization_only=LOCALIZATION_ONLY,
)

# model = RetinaNet(num_classes=pfa_data_module.num_labels + 1)

# load artifact!!

# download checkpoint locally (if not already cached)
run = wandb.init(project=WANDB_PROJECT)
ARTIFACT_NAME = "model-2l5b0myt:v65"
artifact = run.use_artifact(f"ecw/{WANDB_PROJECT}/{ARTIFACT_NAME}", type="model")
artifact_dir = artifact.download()
model = RetinaNet.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")

logger = pl.loggers.WandbLogger(project=WANDB_PROJECT, log_model="all")
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor=MONITOR_ATTRIBUTE, mode="max"
)
earlystop_callback = pl.callbacks.EarlyStopping(
    monitor=MONITOR_ATTRIBUTE, mode="max", patience=20
)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    logger=logger,
    max_epochs=1000,
    callbacks=[checkpoint_callback, earlystop_callback],
)
trainer.fit(model, datamodule=pfa_data_module)
