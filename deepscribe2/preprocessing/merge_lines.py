from typing import List, Tuple

import numpy as np
import torch

from deepscribe2.models.line_detection import SequentialRANSAC
from deepscribe2.utils import get_centroids


def merge_boxes_e2e(
    original_boxes: torch.Tensor, original_labels: List[int]
) -> Tuple[torch.Tensor, List[List[int]]]:
    centroids = get_centroids(original_boxes).numpy()

    ransac_models = SequentialRANSAC().fit(centroids)

    n_merged = np.unique(ransac_models.labels_).shape[0]

    merged_boxes = torch.zeros(n_merged, 4)

    merged_labels = []

    for idx in range(n_merged):
        (line_inds,) = np.where(ransac_models.labels_ == idx)

        line_boxes = original_boxes[line_inds, :]
        merged_boxes[idx, 0] = line_boxes[:, 0].min()
        merged_boxes[idx, 1] = line_boxes[:, 1].min()
        merged_boxes[idx, 2] = line_boxes[:, 2].max()
        merged_boxes[idx, 3] = line_boxes[:, 3].max()

        original_ordering = ransac_models.ordering_[line_inds]

        line_labels = original_labels[line_inds]

        lines_with_ordering = list(zip(line_labels, original_ordering))

        ordered_labels = [
            lab for lab, order in sorted(lines_with_ordering, key=lambda val: val[0])
        ]

        merged_labels.append(ordered_labels)

    return merged_boxes, merged_labels
