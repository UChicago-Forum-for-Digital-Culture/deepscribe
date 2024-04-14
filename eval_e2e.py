import os

import editdistance
import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from deepscribe2.datasets import PFADetectionDataModule
from deepscribe2.pipeline import DeepScribePipeline

# run download_artifacts.sh to download pretrained models!
ARTIFACTS_DIR = "artifacts"
# replace this with wherever you put the data. download_data.sh has an example.
DATA_BASE = "data/DeepScribe_Data_2023-02-04_public"

pfa_datamodule = PFADetectionDataModule(DATA_BASE, batch_size=10)
pfa_datamodule.prepare_data()

pfa_datamodule.setup(stage="test")
# can also initialize these from trained model objects directly.
# run download_artifacts.sh f
pipeline = DeepScribePipeline.from_checkpoints(
    os.path.join(ARTIFACTS_DIR, "detector_epoch=358-step=88673.ckpt"),
    classifier_ckpt=os.path.join(ARTIFACTS_DIR, "classifier_epoch=50-step=2091.ckpt"),
    score_thresh=0.5,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

map_metric = MeanAveragePrecision()
edit_dists = []

failed = 0

with torch.no_grad():
    for imgs, targets in tqdm(pfa_datamodule.test_dataloader()):

        preds = pipeline(imgs)

        preds = [
            {
                key: entry.cpu() if isinstance(entry, torch.Tensor) else entry
                for key, entry in pred.items()
            }
            for pred in preds
        ]
        map_metric.update(preds, targets)
        for pred, targ in zip(preds, targets):
            if "ordering" in pred and pred["ordering"] is not None:
                ordered_labels = pred["labels"][pred["ordering"]].tolist()
                targ_labels = targ["labels"].tolist()
                edit_dist = editdistance.eval(
                    ordered_labels,
                    targ_labels,
                ) / len(targ_labels)
                edit_dists.append(edit_dist)
            else:
                failed += 1
            # compute edit dist

print(
    f"edit dists: {np.median(edit_dists)} / {np.mean(edit_dists)} ({np.std(edit_dists)})"
)

print(f"map: {map_metric.compute()}")

print(f"failed: {failed}")
