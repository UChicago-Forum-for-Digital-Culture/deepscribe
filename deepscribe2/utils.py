from typing import Dict
import torch


def get_boxes(entry: Dict) -> torch.Tensor:
    return torch.tensor(
        [anno["bbox"] for anno in entry["annotations"]], dtype=torch.float
    )


def get_centroids(coords: torch.Tensor):
    output_coords = torch.zeros(coords.size()[0], 2)
    output_coords[:, 0] = (coords[:, 0] + coords[:, 2]) / 2
    output_coords[:, 1] = (coords[:, 1] + coords[:, 3]) / 2
    return output_coords
