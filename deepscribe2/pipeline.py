import warnings
from typing import List

import pandas as pd
import torch
from torch import nn
from torchvision import transforms as T

from deepscribe2.models import ImageClassifier, RetinaNet, SequentialRANSAC
from deepscribe2.transforms import SquarePad
from deepscribe2.utils import get_centroids
from deepscribe2.models.postprocessing import combine_results

warnings.simplefilter(action="ignore", category=FutureWarning)


# is this even needed?
def clamp_boxes(boxes, img_shape):
    # clamp box values
    # 0 < x1, x2 < length
    # 0 < y1, y2 < height
    # torch images are CHW
    boxes[:, 0] = torch.clamp(boxes[:, 0], min=0, max=img_shape[2])
    boxes[:, 2] = torch.clamp(boxes[:, 2], min=0, max=img_shape[2])
    boxes[:, 1] = torch.clamp(boxes[:, 1], min=0, max=img_shape[1])
    boxes[:, 3] = torch.clamp(boxes[:, 3], min=0, max=img_shape[1])


def assign_ordering(preds):
    # get sequence/ordering
    for i, pred in enumerate(preds):
        centroids = get_centroids(pred["boxes"]).numpy()
        try:
            rns = SequentialRANSAC().fit(centroids)
            pred["ordering"] = rns.ordering_.tolist()
            pred["line_assignment"] = rns.labels_.tolist()
        except ValueError as e:
            print(f"RANSAC failed in batch elem {i}")
            pred["ordering"] = None
            pred["line_assignment"] = None


class DeepScribePipeline(nn.Module):
    """
    Pipeline combining separately trained localization and classifier.

    NOTE: no longer storing the class label information here. Pull from the
    dataset.
    """

    def __init__(
        self,
        detector: RetinaNet,
        classifier: ImageClassifier = None,
        score_thresh: float = 0.3,  # apply a score thresh on top of the existing score threshold.
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.detector = detector.to(device)
        self.classifier = classifier.to(device) if classifier else None
        self.score_thresh = score_thresh
        self.device = device

    @classmethod
    def from_checkpoints(
        cls,
        detection_ckpt: str,
        classifier_ckpt: str = None,
        score_thresh: float = 0.3,
        device: str = "cuda",
    ):
        detector = RetinaNet.load_from_checkpoint(
            detection_ckpt, map_location="cpu"
        ).eval()
        classifier = (
            ImageClassifier.load_from_checkpoint(
                classifier_ckpt, map_location="cpu"
            ).eval()
            if classifier_ckpt
            else None
        )

        return cls(
            detector,
            classifier=classifier,
            score_thresh=score_thresh,
            device=device,
        )

    def assign_bbox_labels_classifier(self, preds, imgs):
        if self.classifier is None:
            raise ValueError(
                "No classifier set. Need to provide a classifier to assign bbox labels."
            )

        # get all cutouts from predictions.

        # hack to get around squarepad not supporting tensors.

        transforms = T.Compose(
            [
                T.ToPILImage(),
                SquarePad(),
                T.Resize((50, 50)),
                # Grayscale(num_output_channels=1),
                T.ToTensor(),
            ]
        )
        for img, pred in zip(imgs, preds):
            xformed_cutouts = []
            for i in range(pred["boxes"].shape[0]):
                x1, y1, x2, y2 = pred["boxes"][i, :].long().tolist()
                cutout = img[:, y1:y2, x1:x2]
                xformed_cutouts.append(transforms(cutout))
                # run inference.
                cutouts_tensor = torch.stack(xformed_cutouts).to(self.device)
                cutout_preds = self.classifier(cutouts_tensor)
                # overwrite labels with predictions
                pred["labels"] = cutout_preds.topk(k=1, axis=1).indices.flatten()
                pred["classifier_top5"] = cutout_preds.topk(k=5, axis=1).indices

    # images not needed
    def assign_bbox_labels_multiclass(self, preds):
        return [combine_results(pred) for pred in preds]

    @torch.no_grad()
    def forward(self, imgs: List[torch.Tensor]):
        # run inference with detector
        preds = self.detector([img.to(self.device) for img in imgs])

        # filter preds based oin score thresh
        # clamp preds to images
        for img, pred in zip(imgs, preds):
            pred["boxes"] = pred["boxes"][pred["scores"] > self.score_thresh]
            pred["labels"] = pred["labels"][pred["scores"] > self.score_thresh]
            pred["scores"] = pred["scores"][pred["scores"] > self.score_thresh]
            clamp_boxes(pred["boxes"], img.shape)

        # is a multiclass detector
        if self.detector.hparams.num_classes > 2:
            self.assign_bbox_labels_multiclass(preds)
        else:
            self.assign_bbox_labels_classifier(preds, imgs)

        assign_ordering(preds)

        return preds
