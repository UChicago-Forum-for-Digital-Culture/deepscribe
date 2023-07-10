import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from typing import Dict, Any

from deepscribe2.models.detection.detr import (
    build_position_encoding,
    Backbone,
    Joiner,
    Transformer,
    DETR,
    SetCriterion,
    PostProcess,
    HungarianMatcher,
    NestedTensor,
)
from deepscribe2.models.detection.detr.util import box_ops

from torchmetrics.detection.mean_ap import MeanAveragePrecision


# converts scaled down boxes in xyxywh normalized back
# to original sizes.
def unscale_boxes(boxes, img_w, img_h):
    boxes = box_ops.box_cxcywh_to_xyxy(boxes)
    scale_fct = torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)
    boxes = boxes * scale_fct
    return boxes


class DETRLightningModule(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        num_queries: int = 100,
        aux_loss: bool = False,
        hidden_dim: int = 256,
        backbone_name: str = "resnet50",
        dropout: float = 0.1,
        dilation: bool = False,
        position_embedding: str = "sine",
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
        bbox_loss_coef: float = 5,
        giou_loss_coef: float = 2,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        # build backbone model
        position_enc = build_position_encoding(
            self.hparams.position_embedding, self.hparams.hidden_dim
        )
        train_backbone = self.hparams.lr_backbone > 0
        backbone = Backbone(
            self.hparams.backbone_name,
            train_backbone=train_backbone,
            return_interm_layers=False,
            dilation=self.hparams.dilation,
        )
        backbone_encoded = Joiner(backbone, position_enc)
        backbone_encoded.num_channels = backbone.num_channels

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
            backbone_encoded,
            xformer,
            num_classes=self.hparams.num_classes,
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
            self.hparams.num_classes,
            matcher=self.matcher,
            weight_dict=weight_dict,
            eos_coef=self.hparams.eos_coef,
            losses=losses,
        )

        self.postprocessor = PostProcess()

        self.map = MeanAveragePrecision()

    def init_backbone_from_retinanet_state(self, retinanet_state_dict: Dict[str, Any]):
        # get the matching weights from the retinanet state dict

        backbone_state = {
            key.replace("model.backbone.", ""): val
            for key, val in retinanet_state_dict.items()
            if "backbone" in key and "fpn" not in key  # removing FPN-specific layers
        }

        self.model.backbone[0].load_state_dict(backbone_state, strict=True)

    def forward(self, samples: NestedTensor):
        return self.model(samples)

    def training_step(self, batch, batch_idx):
        images, targets, _ = batch

        outputs = self(images)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        loss = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        self.log_dict(loss_dict, batch_size=images.tensors.size()[0], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets, img_sizes = batch

        # print(targets)
        # raise ValueError()

        outputs = self(images)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        loss = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        self.log(
            "val_loss",
            loss,
            batch_size=images.tensors.size()[0],
        )

        # do we need to filter the labels?
        postprocessed = self.postprocessor(outputs, img_sizes)

        # print(targets[0].keys())

        for elem, (orig_h, orig_w) in zip(targets, img_sizes):
            elem["boxes"] = unscale_boxes(elem["boxes"], img_w=orig_w, img_h=orig_h)

        # print(targets[0]["boxes"])

        # raise ValueError()
        # print(postprocessed[0]["labels"])
        # unsa

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
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.hparams.lr_drop)

        # TODO: set these params dynamically
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=10)
        # cosine_scheduler = CosineAnnealingLR(optimizer, 20)

        # lr_scheduler = SequentialLR(
        #     optimizer, [warmup_scheduler, cosine_scheduler], milestones=[10]
        # )

        return {"optimizer": optimizer, "lr_scheduler": warmup_scheduler}
