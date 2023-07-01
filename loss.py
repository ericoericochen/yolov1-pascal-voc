import torch
import torch.nn as nn
from torchvision.ops import box_iou
import torch.nn.functional as F


class YOLOv1Loss(nn.Module):
    """
    YOLOv1 Loss
    """

    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        """
        S: dimension of the S x S grid
        B: number of bounding boxes predicted by network
        C: number of classes
        lambda_coord: penalty for coord loss
        lambda_noobj: penalty for confidence loss when no object is present in target
        """
        super().__init__()

        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def xywh_to_x1y1x2y2(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Converts YOLO bounding box format to (x1, y1, x2, y2)

        pred: (X, 4)

        returns (X, 4)
        """
        x = boxes[..., 0]  # (N, S^2, X)
        y = boxes[..., 1]
        w = boxes[..., 2]
        h = boxes[..., 3]

        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        x1y1x2y2 = torch.stack((x1, y1, x2, y2), dim=1)

        return x1y1x2y2

    def forward(self, pred, target):
        """
        pred: (N x S x S x (5 * B + C))
        target: (N x S x S x (5 + C))
        """
        print("CALCULATING YOLO LOSS")

        # check pred and target are in the correct shape
        assert len(pred) == len(target)
        N = pred.size(0)

        # get parameters of YOLO loss
        S = self.S
        B = self.B
        C = self.C
        lambda_coord = self.lambda_coord
        lambda_noobj = self.lambda_noobj

        assert pred.shape == torch.Size((N, S, S, 5 * B + C))
        assert target.shape == torch.Size((N, S, S, 5 + C))

        # obj, noobj mask: select bounding boxes whose target bounding box has a confidence=1 for obj and confidence=0
        # for noobj
        obj_mask = target[:, :, :, 0] == 1
        noobj_mask = target[:, :, :, 0] == 0

        # select predictions and targets where ground truth contains an object
        obj_pred = pred[obj_mask]  # (num_obj, 5*B+C)
        obj_target = target[obj_mask]  # (num_obj, 5+C)

        # get bounding boxes
        obj_pred_bndbox = obj_pred[:, : 5 * B].view(
            -1, B, 5
        )  # (num_obj, 5*B+C) -> (num_obj, B, 5)
        obj_target_bndbox = obj_target[:, :5].view(
            -1, 1, 5
        )  # (num_obj, 5*B+C) -> (num_obj, 1, 5)

        # select predictions and targets where ground grouth does not contain an object
        noobj_pred = pred[noobj_mask]  # (num_noobj, 5*B+C)
        noobj_target = target[noobj_mask]  # (num_obj, 5+C)

        # get bounding boxes for target's whose confidenc=0
        noobj_pred_bndbox = noobj_pred[:, : 5 * B].view(
            -1, B, 5
        )  # (num_noobj, 5*B+C) -> (num_noobj, B, 5)
        noobj_target_bndbox = noobj_target[:, :5].view(
            -1, 1, 5
        )  # (num_noobj, 5*B+C) -> (num_noobj, 1, 5)

        # calculate ious
        max_iou_mask = torch.BoolTensor(obj_pred_bndbox.size())
        for i in range(obj_pred_bndbox.size(0)):
            # get proposed boxes and target box
            pred_bndbox = obj_pred_bndbox[i][:, 1:]  # (B, 4)
            target_bndbox = obj_target_bndbox[i][:, 1:]  # (1, 4)

            # convert (x, y, w, h) -> (x1, y1, x2, y2)
            pred_bndbox = self.xywh_to_x1y1x2y2(pred_bndbox)
            target_bndbox = self.xywh_to_x1y1x2y2(target_bndbox)

            # get box ious
            ious = box_iou(pred_bndbox, target_bndbox).squeeze(-1)  # (B)

            # get the box with the max iou and keep in mask
            max_iou, max_idx = ious.max(dim=0)
            max_iou_mask[i, max_idx] = 1

        # responsible predictors
        obj_pred_bndbox = obj_pred_bndbox[max_iou_mask].view(-1, 5)  # (num_obj, 5)
        obj_target_bndbox = obj_target_bndbox.squeeze(1)  # (num_obj, 5)

        ###
        # Bounding Box Loss
        ###
        pred_xy = obj_pred_bndbox[:, 1:3]
        target_xy = obj_target_bndbox[:, 1:3]
        xy_loss = lambda_coord * F.mse_loss(pred_xy, target_xy, reduction="sum")

        pred_wh = torch.sqrt(obj_pred_bndbox[:, 3:5])
        target_wh = torch.sqrt(obj_target_bndbox[:, 3:5])
        wh_loss = lambda_coord * F.mse_loss(pred_wh, target_wh, reduction="sum")

        localization_loss = xy_loss + wh_loss

        ###
        # Confidence Loss
        ###
        obj_pred_confidence = obj_pred_bndbox[:, 0]
        obj_target_confidence = obj_target_bndbox.squeeze(1)[:, 0]
        obj_confidence_loss = F.mse_loss(
            obj_pred_confidence, obj_target_confidence, reduction="sum"
        )

        noobj_pred_confidence = noobj_pred_bndbox[:, :, 0]  # (num_noobj, 2)
        noobj_target_confidence = noobj_target_bndbox[:, :, 0][
            :, [0, 0]
        ]  # (num_noobj, 2) -> duplicated target for every bounding box
        noobj_confidence_loss = lambda_noobj * F.mse_loss(
            noobj_pred_confidence, noobj_target_confidence, reduction="sum"
        )

        confidence_loss = obj_confidence_loss + noobj_confidence_loss

        ###
        # Classification Loss
        ###
        obj_pred_classification = obj_pred[:, -C:]  # (num_obj, C)
        obj_target_classification = obj_target[:, -C:]  # (num_obj, C)
        classification_loss = F.mse_loss(
            obj_pred_classification, obj_target_classification, reduction="sum"
        )

        # total loss
        loss = (localization_loss + confidence_loss + classification_loss) / N

        return loss
