# customizable retinanet head.
import math
from typing import Callable, Optional, List, Dict, Tuple

import torch
from torch import nn, Tensor
from torchvision.ops import boxes as box_ops
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import sigmoid_focal_loss
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection._utils import _box_loss, overwrite_eps
from torchvision.models.detection.retinanet import _v1_to_v2_weights, _sum


class RetinaNetHeadCustomizable(nn.Module):
    """
    A regression and classification head for use in RetinaNet.
    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    """

    def __init__(
        self,
        in_channels,
        num_anchors,
        num_classes,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        prior_probability: float = 0.01,
        fl_gamma: float = 2,
        fl_alpha: float = 0.25,
        reg_loss_type: str = "l1",
        reg_loss_beta: float = 1.0,  # if using smooth l1 loss - 1.0 was default in torchvision but 0.1 in detectron2.
    ):
        super().__init__()
        self.classification_head = RetinaNetClassificationHeadCustomizable(
            in_channels,
            num_anchors,
            num_classes,
            norm_layer=norm_layer,
            prior_probability=prior_probability,
            fl_gamma=fl_gamma,
            fl_alpha=fl_alpha,
        )
        self.regression_head = RetinaNetRegressionHead(
            in_channels,
            num_anchors,
            norm_layer=norm_layer,
            loss_type=reg_loss_type,
            l1_beta=reg_loss_beta,
        )

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Dict[str, Tensor]
        return {
            "classification": self.classification_head.compute_loss(
                targets, head_outputs, matched_idxs
            ),
            "bbox_regression": self.regression_head.compute_loss(
                targets, head_outputs, anchors, matched_idxs
            ),
        }

    def forward(self, x):
        # type: (List[Tensor]) -> Dict[str, Tensor]
        return {
            "cls_logits": self.classification_head(x),
            "bbox_regression": self.regression_head(x),
        }


class RetinaNetClassificationHeadCustomizable(nn.Module):
    """
    A classification head for use in RetinaNet.
    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    """

    _version = 2

    def __init__(
        self,
        in_channels,
        num_anchors,
        num_classes,
        prior_probability=0.01,
        fl_gamma: float = 2,
        fl_alpha: float = 0.25,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(
                misc_nn_ops.Conv2dNormActivation(
                    in_channels, in_channels, norm_layer=norm_layer
                )
            )
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)

        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(
            self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability)
        )

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.fl_alpha = fl_alpha
        self.fl_gamma = fl_gamma

        # This is to fix using det_utils.Matcher.BETWEEN_THRESHOLDS in TorchScript.
        # TorchScript doesn't support class attributes.
        # https://github.com/pytorch/vision/pull/1697#issuecomment-630255584
        self.BETWEEN_THRESHOLDS = det_utils.Matcher.BETWEEN_THRESHOLDS

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            _v1_to_v2_weights(state_dict, prefix)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def compute_loss(self, targets, head_outputs, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Tensor
        losses = []

        cls_logits = head_outputs["cls_logits"]

        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(
            targets, cls_logits, matched_idxs
        ):
            # determine only the foreground
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

            # create the target classification
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target[
                foreground_idxs_per_image,
                targets_per_image["labels"][
                    matched_idxs_per_image[foreground_idxs_per_image]
                ],
            ] = 1.0

            # find indices for which anchors should be ignored
            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

            # compute the classification loss
            losses.append(
                sigmoid_focal_loss(
                    cls_logits_per_image[valid_idxs_per_image],
                    gt_classes_target[valid_idxs_per_image],
                    reduction="sum",
                    alpha=self.fl_alpha,
                    gamma=self.fl_gamma,
                )
                / max(1, num_foreground)
            )

        return _sum(losses) / len(targets)

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_cls_logits = []

        for features in x:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)  # Size=(N, HWA, 4)

            all_cls_logits.append(cls_logits)

        return torch.cat(all_cls_logits, dim=1)


class RetinaNetRegressionHead(nn.Module):
    """
    A regression head for use in RetinaNet.
    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    """

    _version = 2

    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
    }

    def __init__(
        self,
        in_channels,
        num_anchors,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        loss_type: str = "l1",
        l1_beta: float = 0.1,
    ):
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(
                misc_nn_ops.Conv2dNormActivation(
                    in_channels, in_channels, norm_layer=norm_layer
                )
            )
        self.conv = nn.Sequential(*conv)

        self.bbox_reg = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.bbox_reg.weight, std=0.01)
        torch.nn.init.zeros_(self.bbox_reg.bias)

        for layer in self.conv.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.loss_type = loss_type
        self.l1_beta = l1_beta

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            _v1_to_v2_weights(state_dict, prefix)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Tensor
        losses = []

        bbox_regression = head_outputs["bbox_regression"]

        for (
            targets_per_image,
            bbox_regression_per_image,
            anchors_per_image,
            matched_idxs_per_image,
        ) in zip(targets, bbox_regression, anchors, matched_idxs):
            # determine only the foreground indices, ignore the rest
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs_per_image.numel()

            # select only the foreground boxes
            matched_gt_boxes_per_image = targets_per_image["boxes"][
                matched_idxs_per_image[foreground_idxs_per_image]
            ]
            bbox_regression_per_image = bbox_regression_per_image[
                foreground_idxs_per_image, :
            ]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

            # compute the loss
            losses.append(
                _box_loss(
                    self.loss_type,
                    self.box_coder,
                    anchors_per_image,
                    matched_gt_boxes_per_image,
                    bbox_regression_per_image,
                    cnf={"beta": self.l1_beta},
                )
                / max(1, num_foreground)
            )

        return _sum(losses) / max(1, len(targets))

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_bbox_regression = []

        for features in x:
            bbox_regression = self.conv(features)
            bbox_regression = self.bbox_reg(bbox_regression)

            # Permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)  # Size=(N, HWA, 4)

            all_bbox_regression.append(bbox_regression)

        return torch.cat(all_bbox_regression, dim=1)
