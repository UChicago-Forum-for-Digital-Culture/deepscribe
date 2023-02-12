import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader
import wandb

from deepscribe2.datasets.dataset import CuneiformLocalizationDataset, collate_retinanet
from deepscribe2.models.detection import RetinaNet
from deepscribe2 import transforms as T

DATA_BASE = "/local/ecw/DeepScribe_Data_2023-02-04-selected"
WANDB_PROJECT = "deepscribe-torchvision"


imgs_base = f"{DATA_BASE}/cropped_images"

train_file = f"{DATA_BASE}/data_train.json"
val_file = f"{DATA_BASE}/data_val.json"


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


# model = RetinaNet(num_classes=1)

# load artifact!!

# download checkpoint locally (if not already cached)
run = wandb.init(project=WANDB_PROJECT)
artifact = run.use_artifact(f"ecw/{WANDB_PROJECT}/model-vjy1binx:v90", type="model")
artifact_dir = artifact.download()
model = RetinaNet.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")


loader = DataLoader(
    CuneiformLocalizationDataset(
        train_file, imgs_base, transforms=xforms, localization_only=True
    ),
    batch_size=5,
    shuffle=True,
    collate_fn=collate_retinanet,
    num_workers=12,
)
val_loader = DataLoader(
    CuneiformLocalizationDataset(val_file, imgs_base, localization_only=True),
    batch_size=5,
    collate_fn=collate_retinanet,
    num_workers=12,
)

logger = pl.loggers.WandbLogger(project=WANDB_PROJECT, log_model="all")
checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="map_50", mode="max")
# logger = pl.loggers.CSVLogger("logs")

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    logger=logger,
    max_epochs=250,
    callbacks=[checkpoint_callback],
)
trainer.fit(model, train_dataloaders=loader, val_dataloaders=val_loader)
