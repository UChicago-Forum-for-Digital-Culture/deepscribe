from deepscribe2.pipeline import DeepScribePipeline
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from deepscribe2.datasets import PFADetectionDataModule
import editdistance
from pathlib import Path
import wandb
import numpy as np
from tqdm import tqdm

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

pfa_datamodule = PFADetectionDataModule(DATA_BASE, batch_size=512)
pfa_datamodule.setup(stage="test")

pipeline = DeepScribePipeline(
    detection_ckpt, pfa_datamodule.categories_file, classifier_ckpt=classification_ckpt
)

map_metric = MeanAveragePrecision()
edit_dists = []

for imgs, targets in tqdm(pfa_datamodule.test_dataloader()):
    preds = pipeline(imgs)

    map_metric.update(preds, targets)

    for pred, targ in zip(preds, targets):
        edit_dists.append(
            editdistance.eval(
                pred["labels"].flatten()[pred["ordering"]].tolist(),
                targ["labels"].tolist(),
            )
        )
    # compute edit dist

print(
    f"edit dists: {np.median(edit_dists)} / {np.mean(edit_dists)} ({np.std(edit_dists)})"
)

print(f"map: {map_metric.compute()}")
