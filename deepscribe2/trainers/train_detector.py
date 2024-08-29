import pytorch_lightning as pl

from deepscribe2 import transforms as T
from deepscribe2.datasets import PFADetectionDataModule
from deepscribe2.models.detection.retinanet import RetinaNet

DATA_BASE = "data/DeepScribe_Data_2023-02-04_public"
MONITOR_ATTRIBUTE = "map_50"
BSIZE = 3
WANDB_PROJECT = "deepscribe-torchvision"
USE_WANDB = True  # set to false to skip wandb


LOCALIZATION_ONLY = False

xforms = T.Compose(
    [
        T.RandomHorizontalFlip(),
        T.RandomShortestSize(
            [500, 640, 672, 704, 736, 768, 800], 1333
        ),  # taken directly from detectron2 config.
    ]
)


pfa_data_module = PFADetectionDataModule(
    DATA_BASE,
    autocrop=True,
    batch_size=BSIZE,
    train_xforms=xforms,
    localization_only=LOCALIZATION_ONLY,
    start_from_one=True,  # this is required for retinanet to work properly.
)
if USE_WANDB:
    logger = pl.loggers.WandbLogger(project=WANDB_PROJECT, log_model="all")
    # add other hparams
    logger.experiment.config["batch_size"] = BSIZE
    logger.experiment.config["localization_only"] = LOCALIZATION_ONLY
    logger.experiment.config["start_from_one"] = True
    logger.experiment.config["datamodule_labels"] = pfa_data_module.num_labels
else:
    logger = None

print(
    f"training with {pfa_data_module.num_labels} labels, including background: {pfa_data_module.hparams.start_from_one}"
)

model = RetinaNet(num_classes=pfa_data_module.num_labels)


checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor=MONITOR_ATTRIBUTE, mode="max", save_top_k=5
)
lr_callback = pl.callbacks.LearningRateMonitor(
    logging_interval="epoch", log_momentum=False
)
earlystop_callback = pl.callbacks.EarlyStopping(
    monitor=MONITOR_ATTRIBUTE, mode="max", patience=20
)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    logger=logger,
    max_epochs=1000,
    callbacks=[checkpoint_callback, lr_callback, earlystop_callback],
)
trainer.fit(model, datamodule=pfa_data_module)
