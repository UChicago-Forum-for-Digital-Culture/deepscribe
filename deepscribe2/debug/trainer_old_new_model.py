import pytorch_lightning as pl
from torch.utils.data import DataLoader

from deepscribe2.datasets.dataset import CuneiformLocalizationDataset, collate_retinanet
from deepscribe2.models.detection.retinanet import RetinaNet
from deepscribe2 import transforms as T
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


DATA_BASE = "/local/ecw/DeepScribe_Data_2023-02-04-selected"


imgs_base = f"{DATA_BASE}/cropped_images"

train_file = f"{DATA_BASE}/data_train.json"
val_file = f"{DATA_BASE}/data_val.json"

# train_file = f"{DATA_BASE}/data_train_cropped.json"
# val_file = f"{DATA_BASE}/data_val_cropped.json"

categories_file = f"{DATA_BASE}/categories.txt"


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

model = RetinaNet(num_classes=1)

loader = DataLoader(
    CuneiformLocalizationDataset(
        train_file,
        imgs_base,
        categories_file,
        transforms=xforms,
        localization_only=True,
    ),
    batch_size=5,
    collate_fn=collate_retinanet,
    shuffle=True,
    num_workers=12,
)
val_loader = DataLoader(
    CuneiformLocalizationDataset(
        val_file, imgs_base, categories_file, localization_only=True
    ),
    batch_size=5,
    collate_fn=collate_retinanet,
    num_workers=12,
)

logger = pl.loggers.WandbLogger(project="deepscribe-torchvision", log_model="all")
# logger = pl.loggers.CSVLogger("logs")

earlystop = EarlyStopping("map_50", mode="max", patience=15)
lr_callback = pl.callbacks.LearningRateMonitor(
    logging_interval="epoch", log_momentum=False
)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    logger=logger,
    max_epochs=2500,
    callbacks=[earlystop, lr_callback],
)
trainer.fit(model, train_dataloaders=loader, val_dataloaders=val_loader)
