import torch
from torch import nn
import pytorch_lightning as pl
from deepscribe2.models.detr.position_encoding import build_position_encoding
from deepscribe2.models.detr.backbone import Backbone, Joiner
from deepscribe2.models.detr.transformer import Transformer
from deepscribe2.models.detr.detr import DETR, SetCriterion, PostProcess
from deepscribe2.models.detr.matcher import HungarianMatcher
from deepscribe2.models.detr.util.misc import NestedTensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class DETRLightningModule(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        num_queries: int = 100,
        aux_loss: bool = False,
        hidden_dim: int = 256,
        backbone_name: str = "resnet50",
        dropout: float = 0.1,
        dilation: bool = True,
        position_embedding: str = "v3",
        enc_layers: int = 8,
        dec_layers: int = 8,
        pre_norm: bool = True,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        lr: float = 1e-4,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
        lr_drop: int = 200,
        set_cost_class: float = 1,
        set_cost_bbox: float = 5,
        set_cost_giou: float = 2,
        eos_coef: float = 0.1,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        # build backbone model
        position_enc = build_position_encoding(
            self.hparams.position_embedding, self.hparams.hidden_dim
        )
        train_backbone = self.hparams.lr_backbone > 0
        backbone = Backbone(
            self.hparams.backbone_name, train_backbone, False, self.hparams.dilation
        )
        backbone_model = Joiner(backbone, position_enc)
        backbone_model.num_channels = backbone.num_channels

        xformer = Transformer(
            d_model=self.hparams.hidden_dim,
            dropout=self.hparams.dropout,
            nhead=self.hparams.nheads,
            dim_feedforward=self.hparams.dim_feedforward,
            num_encoder_layers=self.hparams.enc_layers,
            num_decoder_layers=self.hparams.dec_layers,
            normalize_before=self.hparams.pre_norm,
            return_intermediate_dec=True,
        )

        # finally, create detr!!

        self.model = DETR(
            backbone,
            xformer,
            num_classes=num_classes + 1,
            num_queries=self.hparams.num_queries,
            aux_loss=self.hparams.aux_loss,
        )

        # build matcher!

        self.matcher = HungarianMatcher(
            cost_class=self.hparams.set_cost_class,
            cost_bbox=self.hparams.set_cost_bbox,
            cost_giou=self.hparams.set_cost_giou,
        )

        weight_dict = {"loss_ce": 1, "loss_bbox": self.hparams.bbox_loss_coef}
        weight_dict["loss_giou"] = self.hparams.giou_loss_coef
        # TODO this is a hack - from original code, nice
        if self.hparams.aux_loss:
            aux_weight_dict = {}
            for i in range(self.hparams.dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes", "cardinality"]

        self.criterion = SetCriterion(
            num_classes,
            matcher=self.matcher,
            weight_dict=weight_dict,
            eos_coef=self.hparams.eos_coef,
            losses=losses,
        )

        self.postprocessors = {"bbox": PostProcess()}

        self.map = MeanAveragePrecision()

    def forward(self, samples: NestedTensor):
        return self.model(samples)

    def training_step(self, batch, batch_idx):
        samples, targets, img_sizes = batch

        outputs = self(samples)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        loss = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        self.log_dict(loss_dict, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets, img_sizes = batch

        outputs = self(images)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        loss = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        self.log("val_loss", loss)

        postprocessed = self.postprocessors["bbox"](outputs, img_sizes)

        self.map.update(postprocessed, targets)

    def validation_epoch_end(self, outs):
        self.log_dict(self.map.compute())
        self.map.reset()

    def configure_optimizers(self):
        # use different LR for backbone
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": self.hparams.lr_backbone,
            },
        ]

        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.hparams.lr_drop)

        return {"optimizer": optimizer, "lr_schedule": lr_scheduler}
