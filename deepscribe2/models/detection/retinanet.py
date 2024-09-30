import torch
from pytorch_lightning import LightningModule
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.retinanet import _default_anchorgen
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn

from deepscribe2.models.detection.retinanet_head import RetinaNetHeadCustomizable


MAP_TO_SAVE = [
    "map",
    "map_50",
    "map_75",
    "map_small",
    "map_medium",
    "map_large",
    "mar_1",
    "mar_10",
    "mar_100",
    "mar_small",
    "mar_medium",
    "mar_large",
]


# NOTE: ZERO IS BACKGROUND
# still seems to work fine in the single-class case with n_classes = 1.
class RetinaNet(LightningModule):
    def __init__(
        self,
        num_classes: int,  # number OF classes INCLUDING BACKGROUND!!!!! They finally fixed the docs.
        score_thresh=0.05,  # differ from detectron configs, but better results this way. Detectron configs were 0.3 and 0.2 respectively.
        nms_thresh=0.5,
        base_lr: float = 1e-4,  # retinanet paper uses 1e-2 but i've never been able to get that to work on this corpus.
        lr_reduce_patience: int = 5,
        weight_decay: float = 0.01,
        # momentum: float = 0.9,
        classification_prior: float = 0.01,
        fl_gamma: float = 2,
        fl_alpha: float = 0.25,
        reg_loss_type: str = "l1",
        reg_loss_beta: float = 1.0,  # if using smooth l1 loss - 1.0 was default in torchvision but 0.1 in detectron2.
        topk_candidates: int = 1000,
        detections_per_img: int = 300,
        optimizer: str = "adamw",
    ):
        super().__init__()
        self.save_hyperparameters()
        # just realized that mutability might not work if all the data pulls from hparams
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates
        self.model = self.make_resnet()
        self.map = MeanAveragePrecision(max_detection_thresholds=[10, 50, 400])

    def make_resnet(self):
        anchor_generator = _default_anchorgen()

        model = retinanet_resnet50_fpn(
            trainable_backbone_layers=5,
            weights_backbone=None,
            score_thresh=self.score_thresh,
            nms_thresh=self.nms_thresh,
            detections_per_img=self.detections_per_img,
            topk_candidates=self.topk_candidates,
        )

        model.head = RetinaNetHeadCustomizable(
            in_channels=model.backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
            num_classes=self.hparams.num_classes,
            prior_probability=self.hparams.classification_prior,
            fl_gamma=self.hparams.fl_gamma,
            fl_alpha=self.hparams.fl_alpha,
            reg_loss_type=self.hparams.reg_loss_type,
            reg_loss_beta=self.hparams.reg_loss_beta,
        )

        return model

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # fasterrcnn takes only images for eval() mode
        preds = self.model(images)
        self.map.update(preds, targets)

    def validation_epoch_end(self, outs):

        map_results = self.map.compute()

        self.log_dict(
            {key: val for key, val in map_results.items() if key in MAP_TO_SAVE}
        )
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
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.hparams.base_lr,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.hparams.base_lr,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.hparams.base_lr,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay,
            )
        scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=self.hparams.lr_reduce_patience
            ),
            "interval": "epoch",
            "name": "reduce_on_plateau",
            "monitor": "map_50",
        }

        return [optimizer], [scheduler_config]
