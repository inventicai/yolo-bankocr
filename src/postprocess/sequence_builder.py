def build_sequence_from_detections(detections, return_confidences=False):
    """
    Build sequence from YOLO detections with optional confidence scores.
    
    Args:
        detections: YOLO detection results
        return_confidences: If True, return confidence scores along with sequence
        
    Returns:
        If return_confidences=False: sequence string
        If return_confidences=True: (sequence, list of confidence scores)
    """
    if not detections or len(detections) == 0:
        return ("", []) if return_confidences else ""
    
    result = detections[0] if isinstance(detections, list) else detections
    boxes = getattr(result, "boxes", [])
    
    items = []
    for b in boxes:
        x_min = b.xyxy[0][0].item()
        cls_id = int(b.cls[0].item())
        label = result.names[cls_id]
        confidence = b.conf[0].item() if hasattr(b, 'conf') else 1.0
        items.append((x_min, label, confidence))
    
    # Sort by x position (left to right)
    items.sort(key=lambda x: x[0])
    
    sequence = ''.join([x[1] for x in items])
    
    if return_confidences:
        confidences = [x[2] for x in items]
        return sequence, confidences
    else:
        return sequence


def build_sequence(yolo_result):
    boxes = getattr(yolo_result, "boxes", [])
    items = []

    for b in boxes:
        x_min = b.xyxy[0][0].item()
        cls_id = int(b.cls[0].item())
        label = yolo_result.names[cls_id]
        items.append((x_min, label))

    items.sort(key=lambda x: x[0])
    return ''.join([x[1] for x in items])
