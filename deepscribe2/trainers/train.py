import pytorch_lightning as pl
from torch.utils.data import DataLoader
from deepscribe2.datasets.dataset import CuneiformLocalizationDataset, collate_retinanet
from deepscribe2.models.detection import RetinaNet

imgs_base = "/local/ecw/DeepScribe_Data_2023-02-04-selected/images"

fold0_file = "/local/ecw/data_nov_2021_fixednumerals/folds/fold0.json"
fold1_file = "/local/ecw/data_nov_2021_fixednumerals/folds/fold1.json"


# TODO: DATA AUG!!

model = RetinaNet(num_classes=2)

loader = DataLoader(
    CuneiformLocalizationDataset(fold0_file, imgs_base, box_only=True),
    batch_size=2,
    collate_fn=collate_retinanet,
    num_workers=12,
)
val_loader = DataLoader(
    CuneiformLocalizationDataset(fold1_file, imgs_base, box_only=True),
    batch_size=2,
    collate_fn=collate_retinanet,
    num_workers=12,
)
trainer = pl.Trainer(
    accelerator="gpu", devices=1, logger=pl.loggers.CSVLogger("logs"), max_epochs=100
)
trainer.fit(model, train_dataloaders=loader, val_dataloaders=val_loader)
