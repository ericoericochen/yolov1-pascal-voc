import torch
import torch.nn as nn
from torchvision.ops import box_iou


class Loss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.mse = nn.MSELoss(reduction="sum")

    def xyxy(self, xywh_tensor, i, j):
        # Convert xywh to xyxy format
        xyxy_tensor = torch.zeros_like(xywh_tensor)
        xyxy_tensor[:, 0] = (j + xywh_tensor[:, 0]) / self.S - xywh_tensor[
            :, 2
        ] / 2  # x1 = x - w/2
        xyxy_tensor[:, 1] = (i + xywh_tensor[:, 1]) / self.S - xywh_tensor[
            :, 3
        ] / 2  # y1 = y - h/2
        xyxy_tensor[:, 2] = (j + xywh_tensor[:, 0]) / self.S + xywh_tensor[
            :, 2
        ] / 2  # x2 = x + w/2
        xyxy_tensor[:, 3] = (i + xywh_tensor[:, 1]) / self.S + xywh_tensor[
            :, 3
        ] / 2  # y2 = y + h/2

        return xyxy_tensor

    def forward(self, pred, target):
        S = self.S
        B = self.B
        C = self.C
        lambda_coord = self.lambda_coord
        lambda_noobj = self.lambda_noobj

        N = pred.size(0)
        pred = pred.view(-1, 5 * B + C)
        target = target.view(-1, 5 + C)

        # masks
        obj_mask = target[..., 0] == 1
        noobj_mask = target[..., 0] == 0

        # get predictions and targets that contain an object, confidence=1
        obj_pred = pred[obj_mask]
        obj_target = target[obj_mask]

        # get predictions and targets that do not contain object, confidence=1
        noobj_pred = pred[noobj_mask]
        noobj_target = target[noobj_mask]

        # select responsible bounding boxes
        obj_pred_bbox = obj_pred[..., :-C].view(-1, B, 5)
        obj_target_bbox = obj_target[..., :-C].view(-1, 1, 5).clone()
        obj_target_bbox_ = obj_target[..., :-C].view(-1, 1, 5).clone()

        grid_indices = torch.arange(0, S * S).to(pred.device)[obj_mask]
        num_obj = obj_target.size(0)
        max_iou_mask = torch.zeros((num_obj, B), dtype=torch.bool)

        for i in range(num_obj):
            grid_index = grid_indices[i]
            pred_bbox = obj_pred_bbox[i][..., 1:].square().sqrt()
            target_bbox = obj_target_bbox_[i][:, 1:5]

            y = grid_index // 7
            x = grid_index % 7

            # convert to xyxy format
            pred_bbox = self.xyxy(pred_bbox, y, x)
            target_bbox = self.xyxy(target_bbox, y, x)

            # get ious
            ious = box_iou(pred_bbox, target_bbox)
            # ious = calculate_iou(pred_bbox, target_bbox)
            max_iou, indices = torch.max(ious, dim=0)
            max_iou, indices = max_iou[0], indices[0]

            max_iou_mask[i][indices] = 1

            obj_target_bbox[i][0][0] = max_iou

        obj_responsible_bbox = obj_pred_bbox[max_iou_mask]

        ###
        # Localization Loss
        ###
        pred_xy = obj_responsible_bbox[..., 1:3]
        target_xy = obj_target_bbox[..., 1:3].squeeze(1)
        xy_loss = self.mse(pred_xy, target_xy)

        pred_wh = obj_responsible_bbox[..., 3:]
        target_wh = obj_target_bbox[..., 3:].squeeze(1)
        pred_wh = (pred_wh**2).sqrt()
        target_wh = (target_wh**2).sqrt()
        wh_loss = self.mse(pred_wh.sqrt(), target_wh.sqrt())

        localization_loss = lambda_coord * (xy_loss + wh_loss)

        ###
        # Confidence Loss
        ###

        # obj: confidence = 1
        obj_pred_confidence = obj_responsible_bbox[..., 0]
        target_ious = obj_target_bbox[..., 0].squeeze(1)
        obj_confidence_loss = self.mse(obj_pred_confidence, target_ious)

        # noobj: confidence = 0
        noobj_pred_bbox = noobj_pred[..., : 5 * B]
        noobj_pred_confidence = noobj_pred_bbox[..., 0::5]
        noobj_target_confidence = torch.zeros_like(noobj_pred_confidence)
        noobj_confidence_loss = lambda_noobj * self.mse(
            noobj_pred_confidence, noobj_target_confidence
        )

        confidence_loss = obj_confidence_loss + noobj_confidence_loss

        ###
        # Classification Loss
        ###
        # print("CLASSIFICATION LOSS")
        pred_classification = obj_pred[..., -C:]
        target_classification = obj_target[..., -C:]
        classification_loss = self.mse(pred_classification, target_classification)

        loss = (localization_loss + confidence_loss + classification_loss) / N

        return loss