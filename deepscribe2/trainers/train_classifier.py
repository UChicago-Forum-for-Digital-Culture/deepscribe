import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T

from deepscribe2.datasets import PFAClassificationDataModule

from deepscribe2.models.classification import ImageClassifier
from deepscribe2.transforms import SquarePad
from deepscribe2.datasets.dataset_folder import HotspotDatasetFolder
from deepscribe2.datasets.direct_dataset import DirectHotspotDataset

DATA_BASE = "/local/ecw/DeepScribe_Data_2023-02-04-selected"
WANDB_PROJECT = "deepscribe-torchvision-classifier"
MONITOR_ATTRIBUTE = "val_Accuracy_top5_micro"

train_xforms = [
    T.RandomAffine(0, translate=(0.2, 0.2)),
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    T.RandomPerspective(),
]

pfa_datamodule = PFAClassificationDataModule(
    DATA_BASE, batch_size=512, hotspot_size=(50, 50), train_xforms=train_xforms
)

model = ImageClassifier(pfa_datamodule.num_labels)

logger = pl.loggers.WandbLogger(project=WANDB_PROJECT)
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor=MONITOR_ATTRIBUTE, mode="max"
)
earlystop_callback = pl.callbacks.EarlyStopping(
    monitor=MONITOR_ATTRIBUTE, mode="max", patience=10
)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    logger=logger,
    max_epochs=10,
    callbacks=[checkpoint_callback, earlystop_callback],
)
trainer.fit(model, datamodule=pfa_datamodule)
