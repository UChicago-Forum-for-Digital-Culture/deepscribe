from pathlib import Path

import pytorch_lightning as pl

import wandb
from deepscribe2 import transforms as T
from deepscribe2.datasets import PFADetectionDataModule
from deepscribe2.models.detection.retinanet_old import RetinaNet

DATA_BASE = "/local/ecw/DeepScribe_Data_2023-02-04-selected"
WANDB_PROJECT = "deepscribe-torchvision"
MONITOR_ATTRIBUTE = "loss"
LOCALIZATION_ONLY = True

xforms = T.Compose(
    [
        T.RandomHorizontalFlip(),
        T.RandomShortestSize(
            [500, 640, 672, 704, 736, 768, 800], 1333
        ),  # taken directly from detectron2 config.
        T.RandomIoUCrop(),
        # T.RandomZoomOut(),
        # T.RandomPhotometricDistort(),
    ]
)

batch_size = 3
start_from_one = False

pfa_data_module = PFADetectionDataModule(
    DATA_BASE,
    autocrop=True,
    batch_size=batch_size,
    train_xforms=xforms,
    localization_only=LOCALIZATION_ONLY,
    start_from_one=start_from_one,  # this is required for retinanet to work properly.
)

print(
    f"training with {pfa_data_module.num_labels} labels, including background: {pfa_data_module.hparams.start_from_one}"
)

model = RetinaNet(
    num_classes=pfa_data_module.num_labels,
    # lr_decay=False,
    # lr_reduce_patience=10,
    # weight_decay=5e-3,
    # nms_thresh=0.5,
)

logger = pl.loggers.WandbLogger(project=WANDB_PROJECT, log_model="all")
# add other hparams
logger.experiment.config["batch_size"] = batch_size
logger.experiment.config["localization_only"] = LOCALIZATION_ONLY
logger.experiment.config["start_from_one"] = start_from_one
logger.experiment.config["datamodule_labels"] = pfa_data_module.num_labels


checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor=MONITOR_ATTRIBUTE, mode="min", save_top_k=5
)
lr_callback = pl.callbacks.LearningRateMonitor(
    logging_interval="epoch", log_momentum=True
)
# local_checkpoint = pl.callbacks.ModelCheckpoint(
#     monitor=MONITOR_ATTRIBUTE, mode="min", save_top_k=1, dirpath="/local/ecw/ckpt_test"
# )
# earlystop_callback = pl.callbacks.EarlyStopping(
#     monitor=MONITOR_ATTRIBUTE, mode="min", patience=20
# )

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    logger=logger,
    max_epochs=1000,
    callbacks=[checkpoint_callback, lr_callback],
)
trainer.fit(model, datamodule=pfa_data_module)
