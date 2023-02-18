import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T

from deepscribe2.models.classification import ImageClassifier
from deepscribe2.transforms import SquarePad
from deepscribe2.datasets.dataset_folder import HotspotDatasetFolder
from deepscribe2.datasets.direct_dataset import DirectHotspotDataset

DATA_BASE = "/local/ecw/DeepScribe_Data_2023-02-04-selected"
WANDB_PROJECT = "deepscribe-torchvision-classifier"
MONITOR_ATTRIBUTE = "val_Accuracy_top5_micro"
N_CLASSES = 141  # TODO: pull this from folder structure

base_transforms = T.Compose([SquarePad(), T.ToTensor(), T.Resize((50, 50))])

train_xforms = T.Compose(
    [
        base_transforms,
        T.RandomAffine(0, translate=(0.2, 0.2)),
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        T.RandomPerspective(),
    ]
)

model = ImageClassifier(N_CLASSES)

imgs_base = f"{DATA_BASE}/cropped_images"

train_file = f"{DATA_BASE}/data_train.json"
val_file = f"{DATA_BASE}/data_val.json"

train_dset = HotspotDatasetFolder(
    DATA_BASE + "/all_hotspots/hotspots_train", N_CLASSES, transform=train_xforms
)

val_dset = HotspotDatasetFolder(
    DATA_BASE + "/all_hotspots/hotspots_val", N_CLASSES, transform=base_transforms
)

loader = DataLoader(
    train_dset,
    batch_size=512,
    shuffle=True,
    num_workers=12,
)

val_loader = DataLoader(
    val_dset,
    batch_size=512,
    num_workers=12,
)

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
trainer.fit(model, train_dataloaders=loader, val_dataloaders=val_loader)
