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
