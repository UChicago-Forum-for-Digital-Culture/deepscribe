import warnings
from typing import List

import pandas as pd
import torch
from torch import nn
from torchvision import transforms as T

from deepscribe2.models import ImageClassifier, RetinaNet, SequentialRANSAC
from deepscribe2.transforms import SquarePad
from deepscribe2.utils import get_centroids

warnings.simplefilter(action="ignore", category=FutureWarning)


class DeepScribePipeline(nn.Module):
    """
    Pipeline combining separately trained localization and classifier.
    """

    def __init__(
        self,
        detection_ckpt: str,
        sign_data: str,
        classifier_ckpt: str = None,
        score_thresh: float = 0.5,  # apply a score thresh on top of the existing score threshold.
    ) -> None:
        super().__init__()

        self.detector = RetinaNet.load_from_checkpoint(detection_ckpt).eval()
        self.classifier = (
            ImageClassifier.load_from_checkpoint(classifier_ckpt).eval()
            if classifier_ckpt
            else None
        )

        self.score_thresh = score_thresh
        # load classes info
        # pandas will interpret the sign "NA" as NaN
        sign_data = pd.read_csv(
            sign_data, na_filter=False, names=["sign", "category_id"]
        )
        self.class_labels = sign_data["sign"].tolist()

    @torch.no_grad()
    def forward(self, imgs: List[torch.Tensor]):
        # run inference with detector
        preds = self.detector(imgs)
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
        # get all cutouts from predictions.
        for img_indx, (img, pred) in enumerate(zip(imgs, preds)):
            # filtering preds
            pred["boxes"] = pred["boxes"][pred["scores"] > self.score_thresh]
            pred["labels"] = pred["labels"][pred["scores"] > self.score_thresh]
            pred["scores"] = pred["scores"][pred["scores"] > self.score_thresh]

            # clamp box values
            # 0 < x1, x2 < length
            # 0 < y1, y2 < height
            # torch images are CHW
            pred["boxes"][:, 0] = torch.clamp(
                pred["boxes"][:, 0], min=0, max=img.shape[2]
            )
            pred["boxes"][:, 2] = torch.clamp(
                pred["boxes"][:, 2], min=0, max=img.shape[2]
            )
            pred["boxes"][:, 1] = torch.clamp(
                pred["boxes"][:, 1], min=0, max=img.shape[1]
            )
            pred["boxes"][:, 3] = torch.clamp(
                pred["boxes"][:, 3], min=0, max=img.shape[1]
            )

            if self.classifier:
                # print("Running inference with classifier...")
                xformed_cutouts = []
                num_preds = pred["boxes"].shape[0]
                for i in range(num_preds):
                    x1, y1, x2, y2 = pred["boxes"][i, :].long().tolist()
                    cutout = img[:, y1:y2, x1:x2]
                    xformed_cutouts.append(transforms(cutout))
                # run inference.
                cutouts_tensor = torch.stack(xformed_cutouts)
                cutout_preds = self.classifier(cutouts_tensor)
                # overwrite labels with predictions
                pred["labels"] = cutout_preds.topk(k=1, axis=1).indices
                pred["classifier_top5"] = cutout_preds.topk(k=5, axis=1).indices

            # get sequence/ordering
            centroids = get_centroids(pred["boxes"]).numpy()

            try:
                rns = SequentialRANSAC().fit(centroids)
                pred["ordering"] = rns.ordering_.tolist()
                pred["line_assignment"] = rns.labels_.tolist()
            except ValueError as e:
                print(f"RANSAC failed in batch elem {img_indx}")
                pred["ordering"] = None
                pred["line_assignment"] = None

        return preds
