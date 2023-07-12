import torch
from torchvision.ops import nms, box_iou

def non_maximum_suppression(pred, iou_threshold=0.5, C=20):
    """
    pred: list of processed yolo output predictions (num boxes, 7)
    (max_confidence, x1, y1, x2, y2, max_probability, class_idx)
    """
    nms_boxes = []
    
    # perform nms on each class independently
    for i in range(C):
        # get all predicted boxes belonging in this class
        boxes = pred[pred[:, -1] == i]
        
        if boxes.size(0) == 0:
            continue
        
        xyxy = boxes[:, 1:5]
        scores = boxes[:, 0]
        nms_indices = nms(xyxy, scores, iou_threshold=iou_threshold)
        nms_bboxes = boxes[nms_indices]
        nms_boxes.append(nms_bboxes)
        
    nms_boxes = torch.cat(nms_boxes)
    return nms_boxes

def get_bboxes(output, 
               confidence_threshold=0.5,
               iou_threshold=0.5,
               S=7, B=2, C=20):
    """
    output: SxSx(5B+C)
    confidence_threshold: select boxes above this threshold
    iou_threshold: nms is applied at this threshold
    """
    bboxes = []
    output = output.view(S*S, -1)
    
    # for each cell, get bounding boxes above confidence threshold
    for cell_idx in range(S * S):
        # get upper left corner of cell [0, S)
        gx = cell_idx % S
        gy = cell_idx // S
        
        cell = output[cell_idx] # (5B + C)
        
        # get probabilities + localization tensor
        probabilities = cell[-C:]
        boxes = cell[:-C]
        
        # get class idx and probability -> select max probability
        class_probability, class_idx = probabilities.max(0)
        
        # get bounding boxes above threshold
        boxes = boxes.view(B, -1)
        boxes = boxes[boxes[:, 0] >= confidence_threshold]
        
        # no bounding boxes have predicted confidence above threshold
        if len(boxes) == 0:
            continue
        
        # get properties of bounding box
        x_c = boxes[:, 1]
        y_c = boxes[:, 2]
        w = boxes[:, 3]
        h = boxes[:, 4]
        
        # convert to yolo format
        x = (gx + x_c) / S
        y = (gy + y_c) / S
        
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        
        # construct bounding box tensor (confidence, x1, y1, x2, y2, class probability, class)
        x1y1x2y2 = torch.stack((x1, y1, x2, y2), dim=1).clamp(0, 1)
        confidences = boxes[:, 0:1]
        num_bboxes = confidences.size(0)
        class_probability = class_probability.repeat(num_bboxes, 1)
        class_idx = class_idx.repeat(num_bboxes, 1)
        bbox = torch.cat((confidences, x1y1x2y2, class_probability, class_idx), dim=1)
        bboxes.append(bbox)
        
    # no bounding boxes found
    if len(bboxes) == 0:
        return torch.empty((0, 7))
        
    # concat bboxes into one tensor
    bboxes = torch.cat(bboxes, dim=0)
    
    # perform nms on bounding boxes to filter down number of bounding boxes
    bboxes = non_maximum_suppression(bboxes, iou_threshold=iou_threshold, C=C)
    
    return bboxes

def average_precision(pred, target, c, iou_threshold=0.5):
    # number of targets
    TP_FN = 0
    N = len(pred)
    TP_FP = [] # will be condensed to one-hot vector
    scores = [] # corresponding score for each TP_FN label
    
    # for each image, label predictions as TP or FP (store in a one-hot vector 1=TP, 0=FP)
    for i in range(N):
        pred_boxes = pred[i]
        target_boxes = target[i]
        
        # get boxes classified in the specified class
        pred_boxes = pred_boxes[pred_boxes[:, -1] == c]
        target_boxes = target_boxes[target_boxes[:, -1] == c]
        
        # continue if TP_FN for this image is 0
        if len(target_boxes) == 0:
            continue
        
        num_TP_FN = target_boxes.size(0)
        num_TP_FP = pred_boxes.size(0)
        TP_FP_vec = torch.zeros(num_TP_FP, device=pred_boxes.device)

        # number of target boxes = TP_FN
        TP_FN += num_TP_FN
        
        # 
        if len(pred_boxes) == 0:
            continue

        # calculate pairwise IOU between predicted boxes and target boxes -> highest IOU above threshold gets classified
        # as TP, duplicate bounding boxes + boxes below IOU threshold are classified as FP
        pred_x1y1x2y2 = pred_boxes[:, 1:5]
        target_x1y1x2y2 = target_boxes[:, 1:5]
        
        ious = box_iou(target_x1y1x2y2, pred_x1y1x2y2)        
        max_ious, max_iou_indices = ious.max(dim=1)        
        for max_iou, max_iou_idx in zip(max_ious, max_iou_indices):
            # this is the best bounding box for each target, label TP if it is above IOU threshold,
            # rest of bounding boxes are duplicates
            if max_iou >= iou_threshold:
                TP_FP_vec[max_iou_idx] = 1
        
        # get scores for each bounding box
        confidences = pred_boxes[:, 0]
        
        # append TP_FP and scores
        TP_FP.append(TP_FP_vec)
        scores.append(confidences)
    
    # no targets; AP = 0
    if len(TP_FP) == 0:
        return 0
    
    # condense TP_FP and scores into one vector
    TP_FP = torch.cat(TP_FP)
    scores = torch.cat(scores)
    
    # sort by score
    _, sorted_score_indices = scores.sort(descending=True)
    TP_FP = TP_FP[sorted_score_indices]
    
    # get cumulative TPs
    TP = torch.cumsum(TP_FP, dim=0)
    
    # calculate precision and recall
    # precision = TP / (TP + FP) (predictions), recall = TP / (TP + FN) (ground truths)
    # P = torch.cumsum(torch.ones_like(TP), dim=0)
    P = torch.arange(1, TP.size(0) + 1, device="cuda")
    
    precision = TP / P
    recall = TP / TP_FN
    
    # add 1 in front of precision and 0 in front of recall
    precision = torch.cat((torch.tensor([1], device="cuda"), precision))
    recall = torch.cat((torch.tensor([0], device="cuda"), recall))
    
    # area under PR-curve
    auc = torch.trapezoid(precision, recall)
    
    return auc.item()

def mean_average_precision(pred, target, iou_threshold=0.5, C=20):
    """
    pred, target: N x bounding boxes x 7
    """
    assert len(pred) == len(target)
        
    total_ap = 0
    
    # for each class, calculate the average precision (AUC of PR curve)
    for i in range(C):
        # print(i)
        ap = average_precision(pred, target, iou_threshold=iou_threshold, c=i)
        # print(ap)
        total_ap += ap
    
    # compute the mean of the APs
    mAP = total_ap / C
    return mAP