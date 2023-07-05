from typing import Dict
import torch
import wandb

from pathlib import Path


def get_boxes(entry: Dict) -> torch.Tensor:
    return torch.tensor(
        [anno["bbox"] for anno in entry["annotations"]], dtype=torch.float
    )


def get_centroids(coords: torch.Tensor):
    output_coords = torch.zeros(coords.size()[0], 2)
    output_coords[:, 0] = (coords[:, 0] + coords[:, 2]) / 2
    output_coords[:, 1] = (coords[:, 1] + coords[:, 3]) / 2
    return output_coords


def load_ckpt_from_wandb(
    artifact: str,
    project: str = "deepscribe-torchvision",
    user: str = "ecw",
    artifact_type="model",
):
    # this will initialize a new run. not sure how to stop that.
    run = wandb.init(project=project)
    artifact = run.use_artifact(f"{user}/{project}/{artifact}", type=artifact_type)
    artifact_dir = artifact.download()
    outpath = Path(artifact_dir) / "model.ckpt"
    return torch.load(outpath), outpath
