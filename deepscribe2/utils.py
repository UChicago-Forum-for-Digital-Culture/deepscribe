from typing import Dict
import torch


def get_boxes(entry: Dict) -> torch.Tensor:
    return torch.tensor(
        [anno["bbox"] for anno in entry["annotations"]], dtype=torch.float
    )
