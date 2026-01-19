from PIL import Image, ImageDraw

def draw_segmenter_boxes(original, seg_results):
    """
    Returns a copy of the original image with segmenter bounding boxes drawn.
    """
    img = original.copy()
    draw = ImageDraw.Draw(img)

    for r in seg_results:
        for box in r.boxes:
            x1, y1, x2, y2 = [int(v.item()) for v in box.xyxy[0]]

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            label = r.names[int(box.cls[0])]
            draw.text((x1, y1 - 12), label, fill="red")

    return img
