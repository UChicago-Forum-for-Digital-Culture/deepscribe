import networkx as nx
import torch
from torchvision.ops import box_iou


# performs essentially cross-class NMS.
# creates graph consisting of overlapping boxes, then pulls connected components.
def combine_results(results, iou_thresh=0.2):
    ious = box_iou(results["boxes"], results["boxes"]).cpu().numpy()
    adjacency_graph = nx.from_numpy_array((ious > iou_thresh))
    overlapping_signs = [
        list(sgns) for sgns in list(nx.connected_components(adjacency_graph))
    ]

    new_boxes = torch.zeros((len(overlapping_signs), 4))
    new_scores = torch.zeros((len(overlapping_signs),))
    new_labels = torch.zeros((len(overlapping_signs),), dtype=torch.int)
    top5_labels = torch.full((len(overlapping_signs), 5), -1, dtype=torch.int)

    for i, comp in enumerate(overlapping_signs):
        # get scores and labels
        component_scores = results["scores"][comp]
        component_labels = results["labels"][comp]
        component_boxes = results["boxes"][comp, :]
        top_idx = component_scores.argmax()
        new_boxes[i, :] = component_boxes[top_idx, :]
        new_scores[i] = component_scores[top_idx]
        new_labels[i] = component_labels[top_idx]

        # collect topk labels - 5 at most

        n_labels = min(5, len(comp))

        top5_labels[i, :n_labels] = component_labels[
            component_scores.topk(n_labels).indices
        ]

    return {
        "boxes": new_boxes,
        "labels": new_labels,
        "scores": new_scores,
        "classifier_top5": top5_labels,
    }
