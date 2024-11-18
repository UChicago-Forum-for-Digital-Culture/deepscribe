import pytorch_lightning as pl
from torchvision import transforms as T
from deepscribe2.datasets.image_thresholding import su

from deepscribe2.datasets import PFAClassificationDataModule
from deepscribe2.models.classification import ImageClassifier

DATA_BASE = "data/DeepScribe_Data_2023-02-04_public"
WANDB_PROJECT = "deepscribe-torchvision-classifier"
MONITOR_ATTRIBUTE = "val_loss"
USE_WANDB = True  # set to false to skip wandb


train_xforms = [
    T.GaussianBlur(kernel_size=5),
    T.RandomAffine(0, translate=(0.2, 0.2)),
    T.RandomPerspective(),
    T.Grayscale(num_output_channels=1),
    T.Lambda(su),
]

pfa_datamodule = PFAClassificationDataModule(
    DATA_BASE, batch_size=512, hotspot_size=(50, 50), train_xforms=train_xforms
)

model = ImageClassifier(pfa_datamodule.num_labels, grayscale=True)

if USE_WANDB:
    logger = pl.loggers.WandbLogger(
        project=WANDB_PROJECT, dir="wandb", save_dir="wandb", log_model="all"
    )
else:
    logger = None
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor=MONITOR_ATTRIBUTE, mode="min", save_top_k=5
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
