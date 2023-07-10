def non_maximum_suppression(pred, iou_threshold=0.5, C=20):
    """
    pred: list of processed yolo output predictions (num boxes, 7)
    (max_confidence, x1, y1, x2, y2, max_probability, class_idx)
    """
    nms_boxes = []
    
    # perform nms on each class independently
    for i in range(C):
        print(i)
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

def process_yolo_output(output, 
                        confidence_threshold=0.5,
                        iou_threshold=0.5, S=7, B=2, C=20):
    """
    pred: (N, S, S, 5B+C)
    Process output of YOLO
    
    Returns: N x Boxes x tensor(confidence, x1, y1, x2, y2, probability, class)
    """
    N = output.size(0)
    assert output.shape == torch.Size([N, S, S, 5 * B + C])
    
    processed_output = []
    output = output.view(-1, S * S, 5 * B + C)
    
    for i in range(N):
        bboxes = []
        for cell_idx in range(S * S):
            yolo = output[i][cell_idx] # (5*B+C)
            boxes = yolo[:-C]
            probabilities = yolo[-C:]
            
            # select responsible box (box with the highest confidence)
            boxes = boxes.view(B, -1)            
            confidences = boxes[:, 0]
            max_confidence, max_confidence_idx = confidences.max(0) 
            responsible_box = boxes[max_confidence_idx]
            
            if max_confidence < confidence_threshold:
                continue
            
            # convert xywh to xyxy
            xywh = responsible_box[1:]
            gx = cell_idx % S
            gy = cell_idx // S
            
            x = xywh[0]
            y = xywh[1]
            w = xywh[2]
            h = xywh[3]
            
            x_c = (gx + x) / S
            y_c = (gy + y) / S

            
            x1 = x_c - w / 2
            y1 = y_c - h / 2
            x2 = x_c + w / 2
            y2 = y_c + h / 2
            
            # get class idx + probability
            max_probability, class_idx = probabilities.max(0)
            
            box = torch.stack([max_confidence, x1, y1, x2, y2, max_probability, class_idx])
            box[1:5] = box[1:5].clamp(0, 1) # make sure xys are between 0 and 1
            bboxes.append(box)
        
        bboxes = torch.stack(bboxes)
        
        # perform NMS on the bboxes
        nms_bboxes = non_maximum_suppression(bboxes, iou_threshold=iou_threshold, C=C)    
        processed_output.append(nms_bboxes)
            
    return processed_output

def process_yolo_target(target, S=7, C=20):
    return process_yolo_output(target, S=S, B=1, C=C)