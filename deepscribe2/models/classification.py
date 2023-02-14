from typing import Any, Optional, Tuple
from itertools import product

import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import timm


from torchmetrics import (
    Accuracy,
    ConfusionMatrix,
    CalibrationError,
    MetricCollection,
    Recall,
    Precision,
    AveragePrecision,
    Specificity,
)


class ImageClassifier(LightningModule):
    def __init__(
        self,
        num_classes: int,
        architecture: str = "resnet50",
        use_pretrained: bool = False,
        learning_rate: float = 0.001,
        # focal_loss=False,
        # loss_weights: Optional[torch.Tensor] = None,
        scheduler_patience: int = 5,
        grayscale: bool = False,
        save_all_confusion: bool = False,
    ):
        super().__init__()
        # setup the model
        self.save_hyperparameters()
        # use timm to create model
        self.model = timm.create_model(
            architecture,
            pretrained=use_pretrained,
            in_chans=1 if grayscale else 3,
            num_classes=num_classes,
        )

        self.loss = nn.CrossEntropyLoss()

        # setup metrics

        metric_dict = {}

        for metric_cls, k, averaging_method in product(
            (Accuracy, Recall, Precision, Specificity),
            (1, 3, 5),
            ("micro", "macro"),
        ):
            metric_dict[
                f"{metric_cls.__name__}_top{k}_{averaging_method}"
            ] = metric_cls(
                task="multiclass",
                num_classes=num_classes,
                top_k=k,
                average=averaging_method,
            )

        # adding additional metrics

        # metric_dict["AP"] = AveragePrecision(num_classes=num_classes)
        metric_dict["ECE"] = CalibrationError(
            num_classes=num_classes, task="multiclass"
        )

        metrics = MetricCollection(metric_dict)

        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

        per_class_metrics = MetricCollection(
            [
                Accuracy(num_classes=num_classes, task="multiclass", average="none"),
                Precision(num_classes=num_classes, task="multiclass", average="none"),
                Recall(num_classes=num_classes, task="multiclass", average="none"),
                Specificity(num_classes=num_classes, task="multiclass", average="none"),
                # F1(num_classes=num_classes, average="none"),
            ]
        )

        self.train_per_class = per_class_metrics.clone(prefix="train_")
        self.val_per_class = per_class_metrics.clone(prefix="val_")

        # create confusion matrices
        # self.confusion_val = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        # self.confusion_train = ConfusionMatrix(
        #     task="multiclass", num_classes=num_classes
        # )

    # tensor of [B, C, H, W]
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.model(imgs)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, targets = batch

        pred = self(images)

        loss = self.loss(pred, targets.long())
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        softmax_pred = F.softmax(pred, dim=1)

        # self.confusion_train(softmax_pred, targets)
        # self.train_per_class(softmax_pred, targets)
        self.train_metrics(softmax_pred, targets)

        self.log_dict(self.train_metrics, on_epoch=True)

        # self.log_dict(self.train_per_class, on_epoch=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        images, targets = batch

        pred = self(images)
        loss = self.loss(pred, targets.long())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        softmax_pred = F.softmax(pred, dim=1)

        self.val_metrics(softmax_pred, targets)
        # self.val_per_class(softmax_pred, targets)
        # self.confusion_val(softmax_pred, targets)

        self.log_dict(self.val_metrics, on_epoch=True)
        # self.log_dict(self.val_per_class, on_epoch=True)

    # def on_validation_epoch_end(self) -> None:
    #     output_dir = self.logger.log_dir + "/per_class/val/"
    #     os.makedirs(output_dir, exist_ok=True)

    #     # logging these here - buggy when using standard log_dict

    #     perclass_outname = (
    #         "val_per_class.pt"
    #         if not self.hparams.save_all_confusion
    #         else f"val_per_class_{self.current_epoch}.pt"
    #     )

    #     torch.save(self.val_per_class.compute(), f"{output_dir}/{perclass_outname}")

    #     self.val_per_class.reset()

    #     outname = (
    #         "confmat.pt"
    #         if not self.hparams.save_all_confusion
    #         else f"confmat_{self.current_epoch}.pt"
    #     )

    #     torch.save(self.confusion_val.compute().cpu(), f"{output_dir}/{outname}")

    # def on_train_epoch_end(self):
    #     output_dir = self.logger.log_dir + "/per_class/train/"
    #     os.makedirs(output_dir, exist_ok=True)

    #     perclass_outname = (
    #         "train_per_class.pt"
    #         if not self.hparams.save_all_confusion
    #         else f"train_per_class_{self.current_epoch}.pt"
    #     )

    #     torch.save(self.train_per_class.compute(), f"{output_dir}/{perclass_outname}")

    #     self.train_per_class.reset()

    #     # outname = (
    #     #     "confmat.pt"
    #     #     if not self.hparams.save_all_confusion
    #     #     else f"confmat_{self.current_epoch}.pt"
    #     # )

    #     # torch.save(self.confusion_train.compute().cpu(), f"{output_dir}/{outname}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        scheduler = ReduceLROnPlateau(
            optimizer=optimizer, mode="min", patience=self.hparams.scheduler_patience
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
