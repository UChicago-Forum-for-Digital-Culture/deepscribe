from typing import Any, Optional

import torch
from pytorch_lightning import LightningModule
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.retinanet import RetinaNetHead, retinanet_resnet50_fpn


class RetinaNet(LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-4,
        num_classes: int = 1,
        backbone: Optional[str] = None,
        score_thresh: float = 0.3,  # from detectron configs
        nms_thresh: float = 0.2,  # from detectron configs
        **kwargs: Any
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.backbone = backbone

        self.model = retinanet_resnet50_fpn(
            trainable_backbone_layers=5,
            weights_backbone=None,
            # score_thresh=score_thresh,
            # nms_thresh=nms_thresh,
            **kwargs
        )

        self.model.head = RetinaNetHead(
            in_channels=self.model.backbone.out_channels,
            num_anchors=self.model.head.classification_head.num_anchors,
            num_classes=num_classes,
        )
        self.map = MeanAveragePrecision()

        self.save_hyperparameters()

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

    # def configure_optimizers(self):
    #     return torch.optim.SGD(
    #         self.model.parameters(),
    #         lr=self.learning_rate,
    #         momentum=0.9,
    #         weight_decay=0.005,
    #     )

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,  # weight_decay=0.005
        )

        reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5
        )
        scheduler_config = {
            "scheduler": reduce_scheduler,
            "interval": "epoch",
            "name": "reduce_on_plateau",
            "monitor": "map_50",
        }

        return [optimizer], [scheduler_config]
