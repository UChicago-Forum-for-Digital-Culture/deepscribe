# finally doing this.
from typing import Any

import torch
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import LinearLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2


class FasterRCNN(LightningModule):
    def __init__(
        self,
        num_classes: int,
        base_lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_factor: float = 0.1,
        warmup_epochs: int = 10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.map = MeanAveragePrecision()

        # build model - using default model builder
        self.model = fasterrcnn_resnet50_fpn_v2(
            num_classes=num_classes, trainable_backbone_layers=5
        )

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # fasterrcnn takes only images for eval() mode
        preds = self.model(images)
        self.map.update(preds, targets)

    def validation_epoch_end(self, outs):
        self.log_dict(self.map.compute())
        self.map.reset()

    def forward(self, x):
        self.model.eval()
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log("loss", loss)
        self.log_dict(loss_dict, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.base_lr,
            weight_decay=self.hparams.weight_decay,
        )

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=self.hparams.warmup_factor,
            total_iters=self.hparams.warmup_epochs,
        )

        return {"optimizer": optimizer, "lr_scheduler": warmup_scheduler}
