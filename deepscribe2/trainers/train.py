import pytorch_lightning as pl
from torch.utils.data import DataLoader
from deepscribe2.datasets.dataset import DeepScribeDataset, collate_retinanet
from deepscribe2.models.detection import RetinaNet


# override for actual useful logging


fold0_file = "/local/ecw/data_nov_2021_fixednumerals/folds/fold0.json"
fold1_file = "/local/ecw/data_nov_2021_fixednumerals/folds/fold1.json"

imgs = "/local/ecw/data_nov_2021_fixednumerals/cropped"

# TODO: DATA AUG!!

model = RetinaNet(num_classes=2)

loader = DataLoader(
    DeepScribeDataset(fold0_file, imgs, box_only=True),
    batch_size=2,
    collate_fn=collate_retinanet,
    num_workers=12,
)
val_loader = DataLoader(
    DeepScribeDataset(fold1_file, imgs, box_only=True),
    batch_size=2,
    collate_fn=collate_retinanet,
    num_workers=12,
)
trainer = pl.Trainer(
    accelerator="gpu", devices=1, logger=pl.loggers.CSVLogger("logs"), max_epochs=100
)
trainer.fit(model, train_dataloaders=loader, val_dataloaders=val_loader)
