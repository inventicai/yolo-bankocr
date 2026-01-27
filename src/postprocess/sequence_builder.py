import numpy as np

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

def build_top_n_sequences(yolo_result, n=2):
    """
    Build multiple sequences using top-N most probable detections for each digit.
    Returns a list of sequences, where:
    - Index 0 is the most probable sequence (top-1)
    - Index 1 is the second most probable sequence (top-2)
    - etc.
    """
    boxes = getattr(yolo_result, "boxes", [])
    if not boxes:
        return [''] * n
    
    # Collect items with position and probabilities
    items = []
    for b in boxes:
        x_min = b.xyxy[0][0].item()
        
        # Get the probability distribution for this detection
        # probs contains probabilities for all classes
        if hasattr(b, 'probs') and b.probs is not None:
            # For classification models, probs contains the full distribution
            probs = b.probs.data.cpu()
            # Get top N class indices sorted by probability
            top_n_indices = probs.argsort(descending=True)[:n]
            top_n_labels = [(yolo_result.names[int(idx)], probs[idx].item()) for idx in top_n_indices]
        else:
            # For detection models, we only have the top class
            # Get confidence and class
            cls_id = int(b.cls[0].item())
            conf = b.conf[0].item()
            label = yolo_result.names[cls_id]
            # For detection, we can only provide the top-1, rest will be duplicates
            top_n_labels = [(label, conf)] * n
        
        items.append((x_min, top_n_labels))
    
    # Sort by x position
    items.sort(key=lambda x: x[0])
    
    # Build N sequences
    sequences = []
    for rank in range(n):
        seq = ''.join([item[1][rank][0] if rank < len(item[1]) else item[1][0][0] for item in items])
        sequences.append(seq)
    
    return sequences

def build_sequences_with_keras(yolo_detection_result, crop_image, keras_classifier):
    """
    Build 2 sequences: Top-1 from YOLO, Top-2 from Keras best prediction.
    Uses batch inference for speed.
    
    Args:
        yolo_detection_result: YOLO result containing bounding boxes of digits
        crop_image: PIL Image of the cropped account number region
        keras_classifier: KerasDigitClassifier instance
        
    Returns:
        Tuple of (yolo_sequence, keras_sequence, probabilities_dict)
    """
    boxes = getattr(yolo_detection_result, "boxes", [])
    if not boxes:
        return '', '', {}
    
    # Extract digits and their positions
    digit_items = []
    digit_images = []
    
    for b in boxes:
        x_min = b.xyxy[0][0].item()
        y_min = b.xyxy[0][1].item()
        x_max = b.xyxy[0][2].item()
        y_max = b.xyxy[0][3].item()
        
        # Get YOLO prediction
        cls_id = int(b.cls[0].item())
        yolo_label = yolo_detection_result.names[cls_id]
        yolo_conf = float(b.conf[0].item())
        
        # Crop the digit from the image
        digit_img = crop_image.crop((x_min, y_min, x_max, y_max))
        
        digit_items.append((x_min, yolo_label, yolo_conf))
        digit_images.append(digit_img)
    
    # Sort by x position (left to right)
    sorted_indices = sorted(range(len(digit_items)), key=lambda i: digit_items[i][0])
    digit_items = [digit_items[i] for i in sorted_indices]
    digit_images = [digit_images[i] for i in sorted_indices]
    
    # Batch inference with Keras
    batch_predictions = keras_classifier.predict_batch(digit_images)
    
    # Build sequences and probabilities
    yolo_sequence = ''
    keras_sequence = ''
    probabilities_dict = {}
    
    for pos, (digit_item, probs) in enumerate(zip(digit_items, batch_predictions)):
        yolo_label = digit_item[1]
        yolo_conf = digit_item[2]
        
        # Get top 3 predictions from Keras
        top3_indices = np.argsort(probs)[::-1][:3]
        top3 = {str(idx): float(probs[idx]) for idx in top3_indices}
        
        # Best Keras prediction
        keras_best = str(top3_indices[0])
        
        yolo_sequence += yolo_label
        keras_sequence += keras_best
        
        # Format: "Position X Detected Y Confidence Z"
        key = f"Position {pos} Detected {yolo_label} Confidence {yolo_conf:.2f}"
        probabilities_dict[key] = top3
    
    return yolo_sequence, keras_sequence, probabilities_dict


def build_threshold_sequences_top_n(yolo_detection_result, crop_image, keras_classifier, top_n=2, threshold_percent=10, max_sequences=100):
    """
    Build multiple sequences using top-N predictions from Keras with a confidence threshold.
    For each digit position, considers the top N predictions from the Keras model that are
    within threshold_percent of the top prediction.
    
    Args:
        yolo_detection_result: YOLO result containing bounding boxes of digits
        crop_image: PIL Image of the cropped account number region
        keras_classifier: KerasDigitClassifier instance
        top_n: Number of top predictions to consider per digit (2 or 3)
        threshold_percent: Percentage threshold (e.g., 10 means alternatives within 10% of top score)
        max_sequences: Maximum number of sequences to generate (to avoid combinatorial explosion)
        
    Returns:
        Tuple of (all_sequences_with_accuracy, digit_alternatives, probabilities_dict)
        - all_sequences_with_accuracy: List of tuples (sequence, probability, digit_accuracy)
        - digit_alternatives: List of lists, each containing (digit, probability) tuples for each position
        - probabilities_dict: Detailed probability information
    """
    boxes = getattr(yolo_detection_result, "boxes", [])
    if not boxes:
        return [], [], {}
    
    # Extract digits and their positions
    digit_items = []
    digit_images = []
    
    for b in boxes:
        x_min = b.xyxy[0][0].item()
        y_min = b.xyxy[0][1].item()
        x_max = b.xyxy[0][2].item()
        y_max = b.xyxy[0][3].item()
        
        # Crop the digit from the image
        digit_img = crop_image.crop((x_min, y_min, x_max, y_max))
        
        digit_items.append(x_min)
        digit_images.append(digit_img)
    
    # Sort by x position (left to right)
    sorted_indices = sorted(range(len(digit_items)), key=lambda i: digit_items[i])
    digit_images = [digit_images[i] for i in sorted_indices]
    
    # Batch inference with Keras
    batch_predictions = keras_classifier.predict_batch(digit_images)
    
    # For each digit position, find top N alternatives within threshold
    digit_alternatives = []
    probabilities_dict = {}
    
    for pos, probs in enumerate(batch_predictions):
        # Get top N indices by probability
        top_n_indices = np.argsort(probs)[::-1][:top_n]
        top_prob = probs[top_n_indices[0]]
        
        # Calculate threshold (e.g., if top is 0.6 and threshold is 10%, accept anything >= 0.54)
        threshold_value = top_prob * (1 - threshold_percent / 100.0)
        
        # Find all alternatives within top N and within threshold
        alternatives = []
        for idx in top_n_indices:
            prob = probs[idx]
            if prob >= threshold_value:
                alternatives.append((str(idx), float(prob)))
        
        digit_alternatives.append(alternatives)
        
        # Store detailed info
        probabilities_dict[f"position_{pos}"] = {
            "top_n_predictions": [(str(idx), float(probs[idx])) for idx in top_n_indices],
            "threshold_value": float(threshold_value),
            "alternatives_used": alternatives
        }
    
    # Generate all possible sequences (combinatorial)
    def generate_sequences_recursive(digit_alts, current_seq="", current_prob=1.0, results=[]):
        if not digit_alts:
            results.append((current_seq, current_prob))
            return
        
        # Limit to prevent explosion
        if len(results) >= max_sequences:
            return
        
        first_digit_alts = digit_alts[0]
        rest = digit_alts[1:]
        
        for digit, prob in first_digit_alts:
            generate_sequences_recursive(rest, current_seq + digit, current_prob * prob, results)
        
        return results
    
    # Generate all sequences
    all_sequences_with_probs = []
    generate_sequences_recursive(digit_alternatives, results=all_sequences_with_probs)
    
    # Sort by probability (highest first)
    all_sequences_with_probs.sort(key=lambda x: x[1], reverse=True)
    
    return all_sequences_with_probs, digit_alternatives, probabilities_dict
    """
    Build multiple sequences using Keras predictions with a confidence threshold.
    Generates all combinations where alternative predictions are within threshold_percent of the top prediction.
    
    Args:
        yolo_detection_result: YOLO result containing bounding boxes of digits
        crop_image: PIL Image of the cropped account number region
        keras_classifier: KerasDigitClassifier instance
        threshold_percent: Percentage threshold (e.g., 10 means alternatives within 10% of top score)
        max_sequences: Maximum number of sequences to generate (to avoid combinatorial explosion)
        
    Returns:
        Tuple of (all_sequences, digit_alternatives, probabilities_dict)
        - all_sequences: List of sequence strings sorted by total probability
        - digit_alternatives: List of lists, each containing (digit, probability) tuples for each position
        - probabilities_dict: Detailed probability information for debugging
    """
    boxes = getattr(yolo_detection_result, "boxes", [])
    if not boxes:
        return [], [], {}
    
    # Extract digits and their positions
    digit_items = []
    digit_images = []
    
    for b in boxes:
        x_min = b.xyxy[0][0].item()
        y_min = b.xyxy[0][1].item()
        x_max = b.xyxy[0][2].item()
        y_max = b.xyxy[0][3].item()
        
        # Crop the digit from the image
        digit_img = crop_image.crop((x_min, y_min, x_max, y_max))
        
        digit_items.append(x_min)
        digit_images.append(digit_img)
    
    # Sort by x position (left to right)
    sorted_indices = sorted(range(len(digit_items)), key=lambda i: digit_items[i])
    digit_images = [digit_images[i] for i in sorted_indices]
    
    # Batch inference with Keras
    batch_predictions = keras_classifier.predict_batch(digit_images)
    
    # For each digit position, find alternatives within threshold
    digit_alternatives = []
    probabilities_dict = {}
    
    for pos, probs in enumerate(batch_predictions):
        # Sort indices by probability
        sorted_indices = np.argsort(probs)[::-1]
        top_prob = probs[sorted_indices[0]]
        
        # Calculate threshold (e.g., if top is 0.6 and threshold is 10%, accept anything >= 0.54)
        threshold_value = top_prob * (1 - threshold_percent / 100.0)
        
        # Find all alternatives within threshold
        alternatives = []
        for idx in sorted_indices:
            prob = probs[idx]
            if prob >= threshold_value:
                alternatives.append((str(idx), float(prob)))
            else:
                # Since sorted, no more will meet threshold
                break
        
        digit_alternatives.append(alternatives)
        
        # Store detailed info
        probabilities_dict[f"position_{pos}"] = {
            "top_prediction": alternatives[0][0],
            "top_probability": alternatives[0][1],
            "threshold_value": float(threshold_value),
            "alternatives": alternatives
        }
    
    # Generate all possible sequences (combinatorial)
    def generate_sequences(digit_alts, current_seq="", current_prob=1.0, results=[]):
        if not digit_alts:
            results.append((current_seq, current_prob))
            return
        
        # Limit to prevent explosion
        if len(results) >= max_sequences:
            return
        
        first_digit_alts = digit_alts[0]
        rest = digit_alts[1:]
        
        for digit, prob in first_digit_alts:
            generate_sequences(rest, current_seq + digit, current_prob * prob, results)
        
        return results
    
    # Generate all sequences
    all_sequences_with_probs = []
    generate_sequences(digit_alternatives, results=all_sequences_with_probs)
    
    # Sort by probability (highest first)
    all_sequences_with_probs.sort(key=lambda x: x[1], reverse=True)
    
    # Extract just the sequences
    all_sequences = [seq for seq, prob in all_sequences_with_probs]
    
    return all_sequences_with_probs, digit_alternatives, probabilities_dict


def calculate_digit_accuracy(sequence: str, ground_truth: str) -> float:
    """
    Calculate digit-level accuracy between a sequence and ground truth.
    
    Args:
        sequence: Predicted sequence string
        ground_truth: Ground truth sequence string
        
    Returns:
        Digit-level accuracy as a percentage (0-100)
    """
    if not ground_truth:
        return 0.0
    
    matches = sum(1 for a, b in zip(sequence, ground_truth) if a == b)
    return (matches / len(ground_truth)) * 100

