import pandas as pd
import torch
from torch import nn
from typing import List
from torchvision import transforms as T
from deepscribe2.transforms import SquarePad


from deepscribe2.models import ImageClassifier, RetinaNet, SequentialRANSAC
from deepscribe2.utils import get_centroids


class DeepScribePipeline(nn.Module):
    """
    Pipeline combining separately trained localization and classifier.
    """

    def __init__(
        self,
        detection_ckpt: str,
        sign_data: str,
        classifier_ckpt: str = None,
    ) -> None:
        super().__init__()

        self.detector = RetinaNet.load_from_checkpoint(detection_ckpt)
        self.classifier = (
            ImageClassifier.load_from_checkpoint(classifier_ckpt)
            if classifier_ckpt
            else None
        )

        # load classes info
        # pandas will interpret the sign "NA" as NaN
        self.sign_data = pd.read_csv(
            sign_data, na_filter=False, names=["sign", "category_id"]
        )

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
        for img, pred in zip(imgs, preds):
            if self.classifier:
                print("Running inference with classifier...")
                xformed_cutouts = []
                num_preds = pred["boxes"].shape(0)
                for i in range(num_preds):
                    x1, y1, x2, y2 = pred["boxes"][i, :].tolist()
                    cutout = img[:, y1:y2, x1:x2]
                    xformed_cutouts.append(transforms(cutout))

                # run inference.
                cutouts_tensor = torch.vstack(xformed_cutouts).to(self.device)
                cutout_preds = self.classifier(cutouts_tensor)
                # overwrite labels with predictions
                pred["labels"] = cutout_preds.topk(k=1, axis=1).indices
                pred["classifier_top5"] = cutout_preds.topk(k=5, axis=1).indices

            # get sequence/ordering
            centroids = get_centroids(pred["boxes"]).to_numpy()

            rns = SequentialRANSAC().fit(centroids)
            pred["ordering"] = rns.ordering_.tolist()
            pred["line_assignment"] = rns.labels_.tolist()

        return preds
