import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T

from deepscribe2.models.classification import ImageClassifier

DATA_BASE = "/local/ecw/DeepScribe_Data_2023-02-04-selected"
WANDB_PROJECT = "deepscribe-torchvision-classifier"


base_transforms = T.Compose([T.ToTensor(), T.Resize((100, 100))])

train_xforms = T.Compose(
    [base_transforms, T.RandomPerspective(), T.ColorJitter(brightness=0.5, hue=0.3)]
)

model = ImageClassifier(141)

loader = DataLoader(
    ImageFolder(DATA_BASE + "/all_hotspots/hotspots_train", transform=train_xforms),
    batch_size=1024,
    shuffle=True,
    num_workers=12,
)

val_loader = DataLoader(
    ImageFolder(DATA_BASE + "/all_hotspots/hotspots_val", transform=base_transforms),
    batch_size=1024,
    num_workers=12,
)

logger = pl.loggers.WandbLogger(project=WANDB_PROJECT)
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="val_Accuracy_top5_macro", mode="max"
)

# logger = pl.loggers.CSVLogger("logs")

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    logger=logger,
    max_epochs=100,
    callbacks=[checkpoint_callback],
)
trainer.fit(model, train_dataloaders=loader, val_dataloaders=val_loader)
