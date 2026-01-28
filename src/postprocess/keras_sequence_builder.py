"""
Helper functions to build sequences from bounding boxes using different models.
"""
from PIL import Image
import numpy as np


def extract_digit_crops_from_boxes(image: Image.Image, yolo_result) -> list[tuple[float, Image.Image]]:
    """
    Extract individual digit crops from YOLO bounding boxes.
    
    Args:
        image: PIL Image of the cropped account number region
        yolo_result: YOLO detection result object
        
    Returns:
        List of tuples (x_position, cropped_digit_image)
    """
    boxes = getattr(yolo_result, "boxes", [])
    crops = []
    
    for b in boxes:
        # Get bounding box coordinates (xyxy format)
        x_min = int(b.xyxy[0][0].item())
        y_min = int(b.xyxy[0][1].item())
        x_max = int(b.xyxy[0][2].item())
        y_max = int(b.xyxy[0][3].item())
        
        # Crop the digit from the image
        digit_crop = image.crop((x_min, y_min, x_max, y_max))
        
        # Store with x position for sorting
        crops.append((float(x_min), digit_crop))
    
    # Sort by x position (left to right)
    crops.sort(key=lambda x: x[0])
    
    return crops


async def build_sequence_with_keras(image: Image.Image, yolo_result, keras_model) -> str:
    """
    Build digit sequence using Keras model for classification.
    
    Args:
        image: PIL Image of the cropped account number region
        yolo_result: YOLO detection result with bounding boxes
        keras_model: KerasDigitClassifier instance
        
    Returns:
        Predicted sequence string
    """
    # Extract crops from bounding boxes
    crops = extract_digit_crops_from_boxes(image, yolo_result)
    
    if not crops:
        return ""
    
    # Get just the images in order
    digit_images = [crop_img for _, crop_img in crops]
    
    # Predict all digits at once
    predicted_digits = await keras_model.predict_batch(digit_images)
    
    # Build sequence string
    sequence = ''.join(str(digit) for digit in predicted_digits)
    
    return sequence


def build_sequence(yolo_result):
    """
    Build sequence using YOLO's own classification (original method).
    
    Args:
        yolo_result: YOLO detection result object
        
    Returns:
        Predicted sequence string
    """
    boxes = getattr(yolo_result, "boxes", [])
    items = []

    for b in boxes:
        x_min = b.xyxy[0][0].item()
        cls_id = int(b.cls[0].item())
        label = yolo_result.names[cls_id]
        items.append((x_min, label))

    items.sort(key=lambda x: x[0])
    return ''.join([x[1] for x in items])


def build_keras_sequence(keras_model, digit_crops, return_confidences=False):
    """
    Build sequence using Keras model for digit classification.
    
    Args:
        keras_model: KerasDigitClassifier instance
        digit_crops: List of PIL Image crops of digits
        return_confidences: If True, return confidence scores along with sequence
        
    Returns:
        If return_confidences=False: sequence string
        If return_confidences=True: (sequence, list of confidence scores)
    """
    if not digit_crops:
        return ("", []) if return_confidences else ""
    
    if return_confidences:
        # Use method that returns confidences
        predicted_digits, confidences = keras_model.predict_with_confidence(digit_crops)
        sequence = ''.join(str(digit) for digit in predicted_digits)
        return sequence, confidences
    else:
        # Use regular sync prediction
        predicted_digits = keras_model._predict_batch_sync(digit_crops)
        sequence = ''.join(str(digit) for digit in predicted_digits)
        return sequence
