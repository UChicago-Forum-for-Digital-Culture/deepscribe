# derived from pl_bolts.

from typing import Any, Optional

import torch
from torch import nn
from pytorch_lightning import LightningModule
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.retinanet import (
    _default_anchorgen,
    resnet50,
    RetinaNet as torchvision_retinanet,
)
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor

from deepscribe2.models.detection.retinanet_head import (
    RetinaNetHeadCustomizable,
)

# from torchvision.models.detection.anchor_utils import AnchorGenerator


# NOTE: ZERO IS BACKGROUND
class RetinaNet(LightningModule):
    def __init__(
        self,
        num_classes: int,  # number OF classes INCLUDING BACKGROUND!!!!! They finally fixed the docs.
        # backbone: Optional[str] = None,
        score_thresh: float = 0.3,  # from detectron configs
        nms_thresh: float = 0.2,  # from detectron configs
        base_lr: float = 1e-4,  # retinanet paper uses 1e-2 but i've never been able to get that to work on this corpus.
        lr_decay: bool = True,
        lr_decay_step: int = 50,
        lr_decay_gamma: float = 0.1,
        lr_reduce_patience: Optional[int] = 10,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        classification_prior: float = 0.01,
        fl_gamma: float = 2,
        fl_alpha: float = 0.25,
        reg_loss_type: str = "smooth_l1",
        reg_loss_beta: float = 0.1,  # if using smooth l1 loss - 1.0 was default in torchvision but 0.1 in detectron2.
        topk_candidates: int = 1000,
        detections_per_img=300,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = self.make_resnet()

        self.map = MeanAveragePrecision()

    def make_resnet(self):
        # configs the resnet.
        backbone = resnet50(weights=None, progress=False, norm_layer=nn.BatchNorm2d)
        backbone = _resnet_fpn_extractor(
            backbone,
            5,
            returned_layers=[2, 3, 4],
            extra_blocks=LastLevelP6P7(256, 256),
        )
        anchor_generator = _default_anchorgen()

        head = RetinaNetHeadCustomizable(
            in_channels=backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
            num_classes=self.hparams.num_classes,
            prior_probability=self.hparams.classification_prior,
            fl_gamma=self.hparams.fl_gamma,
            fl_alpha=self.hparams.fl_alpha,
            reg_loss_type=self.hparams.reg_loss_type,
            reg_loss_beta=self.hparams.reg_loss_beta,
        )

        model = torchvision_retinanet(
            backbone,
            self.hparams.num_classes,
            nms_thresh=self.hparams.nms_thresh,
            score_thresh=self.hparams.score_thresh,
            head=head,
            topk_candidates=self.hparams.topk_candidates,
            detections_per_img=self.hparams.detections_per_img,
        )

        return model

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

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.base_lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )

        lr_schedulers = []

        if self.hparams.lr_decay:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, self.hparams.lr_decay_step, gamma=self.hparams.lr_decay_gamma
            )
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",
                "name": "stepLR",
            }

            lr_schedulers.append(scheduler_config)

        if self.hparams.lr_reduce_patience:
            reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max"
            )
            scheduler_config = {
                "scheduler": reduce_scheduler,
                "interval": "epoch",
                "name": "reduce_on_plateau",
                "monitor": "map_50",
            }

        return [optimizer], lr_schedulers
