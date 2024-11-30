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

    top1_truepositive = (matched_true_cls == pred_cls)[~false_positive].float()

    metrics = {
        "fpr": float(false_positive.float().mean()),
        "fp_count": float(false_positive.sum()),
        "count": false_positive.shape[0],
        "top1_acc": float(top1_truepositive.mean()),
        "top1_truepositive": float(top1_truepositive.sum()),
    }

    if "classifier_top5" in preds:
        has_match = (
            torch.eq(matched_true_cls.unsqueeze(1), preds["classifier_top5"])
            .float()
            .sum(1)
            > 0
        ).float()
        metrics["top5_acc"] = float(has_match[~false_positive].mean())
        metrics["top5_truepositive"] = float(has_match[~false_positive].sum())

    return metrics


def compute_cls_metrics_agged(labels_all, preds_all, iou_thresh=0.5):

    count = 0
    false_positives = 0
    top1_tp = 0
    top5_tp = 0

    for labels, preds in zip(labels_all, preds_all):
        metrics = compute_cls_metrics(labels, preds, iou_thresh=iou_thresh)
        count += metrics["count"]
        false_positives += metrics["fp_count"]
        top1_tp += metrics["top1_truepositive"]
        top5_tp += metrics["top5_truepositive"]

    results = {
        "fpr": false_positives / count,
        "top1_acc": top1_tp / count,
        "top5_acc": top5_tp / count,
    }

    return results
