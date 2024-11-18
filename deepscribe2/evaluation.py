from torchvision.ops import box_iou
import numpy as np
import torch


# compute classification-style stats for the results.
# match predicted boxes to the best possible match for a true box,
# then compute topk accuracies for match.
# this is slightly generous in the sense that if a predicted box matches with
# more than one true box, it won't get penalized.
def compute_cls_metrics(labels, preds, iou_thresh=0.5):

    true_boxes, true_cls = labels["boxes"], labels["labels"]
    pred_boxes, pred_cls = preds["boxes"], preds["labels"]

    # n_true x n_pred
    ious = box_iou(true_boxes, pred_boxes).cpu()
    # n_pred x 1
    best_match = ious.argmax(0)
    # n_pred x 1
    best_ious = ious[best_match, np.arange(ious.shape[1])]

    false_positive = best_ious < iou_thresh

    matched_true_cls = true_cls[best_match]

    top1_acc = (matched_true_cls == pred_cls)[~false_positive].float().mean()

    metrics = {"fpr": false_positive.float().mean(), "top1_acc": top1_acc}

    if "classifier_top5" in preds:
        has_match = (
            torch.eq(matched_true_cls.unsqueeze(1), preds["classifier_top5"])
            .float()
            .sum(1)
        )
        metrics["top5_acc"] = has_match[~false_positive].mean()

    return metrics
