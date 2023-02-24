import pytorch_lightning as pl
from torchvision import transforms as T

from deepscribe2.datasets import PFAClassificationDataModule
from deepscribe2.models.classification import ImageClassifier

DATA_BASE = "/local/ecw/DeepScribe_Data_2023-02-04-selected"
WANDB_PROJECT = "deepscribe-torchvision-classifier"
MONITOR_ATTRIBUTE = "val_loss"

train_xforms = [
    T.RandomAffine(0, translate=(0.2, 0.2)),
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    T.RandomPerspective(),
]

pfa_datamodule = PFAClassificationDataModule(
    DATA_BASE, batch_size=512, hotspot_size=(50, 50), train_xforms=train_xforms
)

model = ImageClassifier(pfa_datamodule.num_labels)

logger = pl.loggers.WandbLogger(project=WANDB_PROJECT, log_model="all")
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor=MONITOR_ATTRIBUTE, mode="min"
)
earlystop_callback = pl.callbacks.EarlyStopping(
    monitor=MONITOR_ATTRIBUTE, mode="min", patience=20
)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    logger=logger,
    max_epochs=100,
    callbacks=[checkpoint_callback, earlystop_callback],
)
trainer.fit(model, datamodule=pfa_datamodule)
