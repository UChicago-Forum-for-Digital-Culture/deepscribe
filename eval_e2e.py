from deepscribe2.pipeline import DeepScribePipeline
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from deepscribe2.datasets import PFADetectionDataModule
import editdistance
from pathlib import Path
import wandb
import numpy as np
from tqdm import tqdm
import torch

# download checkpoint locally (if not already cached)
run = wandb.init(project="deepscribe-e2e")
artifact = run.use_artifact(
    f"ecw/deepscribe-torchvision/model-vjy1binx:v90", type="model"
)
artifact_dir = artifact.download()
detection_ckpt = Path(artifact_dir) / "model.ckpt"

artifact = run.use_artifact(
    f"ecw/deepscribe-torchvision-classifier/model-7mjv8mn7:v19", type="model"
)
artifact_dir = artifact.download()
classification_ckpt = Path(artifact_dir) / "model.ckpt"

DATA_BASE = "/local/ecw/DeepScribe_Data_2023-02-04-selected"

pfa_datamodule = PFADetectionDataModule(DATA_BASE, batch_size=10)
pfa_datamodule.setup(stage="test")

pipeline = DeepScribePipeline(
    detection_ckpt,
    pfa_datamodule.categories_file,
    classifier_ckpt=classification_ckpt,
    score_thresh=0.5,
)
# use_cuda = torch.cuda.is_available()
use_cuda = False
if use_cuda:
    pipeline.cuda()

map_metric = MeanAveragePrecision()
edit_dists = []

failed = 0

# i still cannot figure out the scope of torch.no_grad.
with torch.no_grad():
    for imgs, targets in tqdm(pfa_datamodule.test_dataloader()):
        if use_cuda:
            imgs = [img.cuda() for img in imgs]
        preds = pipeline(imgs)
        if use_cuda:
            preds = {key: tns.cpu() for key, tns in preds.items()}
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
