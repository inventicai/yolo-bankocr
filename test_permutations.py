"""
Test script that generates permutations from YOLO and Keras predictions.
For digits where models disagree, if confidence threshold is met, 
generate all permutations and test for improved accuracy.
"""

import json
import os
from itertools import product
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from src.models.yolo_wrapper import YoloWrapper
from src.models.keras_wrapper import KerasDigitClassifier
from src.helpers.crop_utils import get_cropped_image_async
from src.helpers.image_io import open_image
from src.postprocess.sequence_builder import build_sequence_from_detections
from src.postprocess.keras_sequence_builder import build_keras_sequence, extract_digit_crops_from_boxes
import asyncio

# Configuration
CONFIDENCE_THRESHOLD = 0.6  # Adjust this threshold as needed
MAX_PERMUTATIONS = 1024  # Safety limit to avoid explosion
TEST_DIR = Path("All_data_png")
OUTPUT_FILE = "permutation_test_results.json"

def load_ground_truth(txt_file: Path):
    """Load ground truth from generated_account.txt file."""
    ground_truth = {}
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip header lines and empty lines
            if not line or line.startswith('=') or line.startswith('Generated') or line.startswith('Template'):
                continue
            # Parse lines like: test_form_0001.png: 833447588304
            if ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    filename = parts[0].strip()
                    account_num = parts[1].strip()
                    ground_truth[filename] = account_num
    return ground_truth

def load_models():
    """Load YOLO and Keras models"""
    print("Loading models...")
    segmenter = YoloWrapper("src/models/segmenter.pt")
    digit_detector = YoloWrapper("src/models/digit_classifier.pt")
    keras_model = KerasDigitClassifier("src/models/DigitClassifier.keras")
    return segmenter, digit_detector, keras_model

async def get_predictions_with_confidence(image_path, segmenter, digit_detector, keras_model):
    """
    Get predictions from both models with confidence scores for each digit.
    Returns: (yolo_sequence, keras_sequence, yolo_confidences, keras_confidences)
    """
    try:
        # Load image
        pil_img = await open_image(image_path)
        
        # Run segmenter to find account number area
        seg_results = await segmenter.predict(pil_img, conf=0.25)
        if not seg_results or len(seg_results) == 0:
            return None, None, None, None
        
        # Crop to account number region
        crop = await get_cropped_image_async(pil_img, segmenter, seg_results)
        if crop is None:
            return None, None, None, None
        
        # Detect individual digits within the cropped area
        digit_results = await digit_detector.predict(crop, conf=0.25)
        if not digit_results or len(digit_results) == 0:
            return None, None, None, None
        
        # Get YOLO sequence with confidences
        yolo_sequence, yolo_confidences = build_sequence_from_detections(
            digit_results, return_confidences=True
        )
        
        # Get Keras sequence with confidences
        digit_crops = extract_digit_crops_from_boxes(crop, digit_results[0])
        digit_images = [img for _, img in digit_crops]
        keras_sequence, keras_confidences = build_keras_sequence(
            keras_model, digit_images, return_confidences=True
        )
        
        return yolo_sequence, keras_sequence, yolo_confidences, keras_confidences
        
    except Exception as e:
        print(f"  Error processing: {e}")
        return None, None, None, None

def find_disagreement_positions(yolo_seq, keras_seq, yolo_conf, keras_conf, threshold):
    """
    Find positions where YOLO and Keras disagree and meet confidence threshold criteria.
    Returns: list of (position, yolo_digit, keras_digit, yolo_conf, keras_conf)
    """
    disagreements = []
    min_len = min(len(yolo_seq), len(keras_seq))
    
    for i in range(min_len):
        if yolo_seq[i] != keras_seq[i]:
            # Get confidence scores
            y_conf = yolo_conf[i] if i < len(yolo_conf) else 0.0
            k_conf = keras_conf[i] if i < len(keras_conf) else 0.0
            
            # Check if uncertainty is high enough to warrant permutation
            # Both should have decent confidence, but neither should be too certain
            if (y_conf >= threshold and k_conf >= threshold and 
                abs(y_conf - k_conf) < 0.3):  # Similar confidence levels
                disagreements.append((i, yolo_seq[i], keras_seq[i], y_conf, k_conf))
    
    return disagreements

def generate_permutations(base_sequence, disagreements, max_perms=MAX_PERMUTATIONS):
    """
    Generate all permutations by replacing disagreement positions with alternatives.
    Returns: list of permutation strings
    """
    if not disagreements:
        return [base_sequence]
    
    # Calculate number of permutations
    num_perms = 2 ** len(disagreements)
    if num_perms > max_perms:
        print(f"  Warning: Too many permutations ({num_perms}), limiting to {max_perms}")
        # Use only the most uncertain disagreements
        disagreements = sorted(disagreements, key=lambda x: abs(x[3] - x[4]))[:10]
        num_perms = 2 ** len(disagreements)
    
    # Generate all combinations
    positions = [d[0] for d in disagreements]
    alternatives = [(d[1], d[2]) for d in disagreements]  # (yolo_digit, keras_digit)
    
    permutations = []
    for choice_tuple in product([0, 1], repeat=len(disagreements)):
        seq_list = list(base_sequence)
        for pos_idx, choice in enumerate(choice_tuple):
            pos = positions[pos_idx]
            digit = alternatives[pos_idx][choice]  # 0=YOLO, 1=Keras
            if pos < len(seq_list):
                seq_list[pos] = digit
        permutations.append(''.join(seq_list))
    
    return permutations

def calculate_digit_accuracy(predicted, actual):
    """Calculate digit-level accuracy"""
    if not predicted and not actual:
        return 1.0
    
    # Pad shorter sequence with empty spaces for comparison
    max_len = max(len(predicted), len(actual))
    pred_padded = predicted.ljust(max_len, ' ')
    act_padded = actual.ljust(max_len, ' ')
    
    correct_digits = sum(1 for p, a in zip(pred_padded, act_padded) if p == a and a != ' ')
    total_digits = len(actual)  # Use actual length as denominator
    
    accuracy = correct_digits / total_digits if total_digits > 0 else 0.0
    return accuracy

def test_with_permutations():
    """Main testing function with permutation generation"""
    # Load ground truth from the same file as test_accuracy.py
    gt_file = TEST_DIR / "generated_account.txt"
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
    
    print("[+] Loading ground truth...")
    actual_lookup = load_ground_truth(gt_file)
    print(f"[+] Loaded {len(actual_lookup)} ground truth entries\n")
    
    # Load models
    segmenter, digit_detector, keras_model = load_models()
    
    # Results storage
    results = []
    failed_images = []  # Track failed images
    summary_stats = {
        'total_tested': 0,
        'permutations_improved': 0,
        'no_disagreements': 0,
        'too_many_permutations': 0,
        'avg_permutations_generated': 0,
        'yolo_original_correct': 0,
        'best_permutation_correct': 0,
        'improvement_cases': [],
        'failed_images': 0  # Add counter
    }
    
    total_permutations = 0
    
    # Get test images (limit to first 100)
    test_files = sorted([f for f in TEST_DIR.glob("test_form_*.png")])[:100]
    
    print(f"Testing {len(test_files)} images with permutation generation...")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Max permutations per image: {MAX_PERMUTATIONS}\n")
    
    # Create event loop
    async def process_all():
        nonlocal total_permutations
        
        for idx, img_path in enumerate(test_files, 1):
            filename = img_path.name
            actual = actual_lookup.get(filename, "")
            
            # Increment total_tested for every image we attempt
            summary_stats['total_tested'] += 1
            
            if not actual:
                print(f"[{idx}/{len(test_files)}] {filename}: No ground truth found, skipping")
                failed_images.append(f"{filename} (no ground truth)")
                summary_stats['failed_images'] += 1
                continue
            
            print(f"[{idx}/{len(test_files)}] Processing {filename}...")
            
            # Get predictions with confidence
            yolo_seq, keras_seq, yolo_conf, keras_conf = await get_predictions_with_confidence(
                img_path, segmenter, digit_detector, keras_model
            )
            
            if yolo_seq is None:
                print(f"  Failed to process")
                failed_images.append(filename)
                summary_stats['failed_images'] += 1
                continue
            
            # Find disagreements
            disagreements = find_disagreement_positions(
                yolo_seq, keras_seq, yolo_conf, keras_conf, CONFIDENCE_THRESHOLD
            )
            
            print(f"  YOLO: {yolo_seq}")
            print(f"  Keras: {keras_seq}")
            print(f"  Actual: {actual}")
            print(f"  Disagreements meeting threshold: {len(disagreements)}")
            
            if not disagreements:
                summary_stats['no_disagreements'] += 1
                results.append({
                    'filename': filename,
                    'actual': actual,
                    'yolo_original': yolo_seq,
                    'keras_original': keras_seq,
                    'disagreements': 0,
                    'permutations_tested': 1,
                    'best_permutation': yolo_seq,
                    'best_accuracy': calculate_digit_accuracy(yolo_seq, actual),
                    'yolo_accuracy': calculate_digit_accuracy(yolo_seq, actual),
                    'improved': False
                })
                continue
            
            # Generate permutations
            permutations = generate_permutations(yolo_seq, disagreements)
            total_permutations += len(permutations)
            
            print(f"  Generated {len(permutations)} permutations")
            
            # Test all permutations
            best_perm = yolo_seq
            best_accuracy = calculate_digit_accuracy(yolo_seq, actual)
            yolo_original_accuracy = best_accuracy
            
            for perm in permutations:
                acc = calculate_digit_accuracy(perm, actual)
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_perm = perm
            
            improved = best_accuracy > yolo_original_accuracy
            
            if improved:
                summary_stats['permutations_improved'] += 1
                summary_stats['improvement_cases'].append({
                    'filename': filename,
                    'yolo_original': yolo_seq,
                    'best_permutation': best_perm,
                    'yolo_accuracy': yolo_original_accuracy,
                    'best_accuracy': best_accuracy,
                    'improvement': best_accuracy - yolo_original_accuracy
                })
                print(f"  ✓ IMPROVED! {yolo_original_accuracy:.2%} -> {best_accuracy:.2%}")
                print(f"    Best: {best_perm}")
            
            # Track sequence-level accuracy
            if yolo_seq == actual:
                summary_stats['yolo_original_correct'] += 1
            if best_perm == actual:
                summary_stats['best_permutation_correct'] += 1
            
            results.append({
                'filename': filename,
                'actual': actual,
                'yolo_original': yolo_seq,
                'keras_original': keras_seq,
                'disagreements': len(disagreements),
                'disagreement_details': [
                    {
                        'position': d[0],
                        'yolo_digit': d[1],
                        'keras_digit': d[2],
                        'yolo_confidence': float(d[3]),
                        'keras_confidence': float(d[4])
                    }
                    for d in disagreements
                ],
                'permutations_tested': len(permutations),
                'best_permutation': best_perm,
                'best_accuracy': best_accuracy,
                'yolo_accuracy': yolo_original_accuracy,
                'improved': improved,
                'sequence_correct': best_perm == actual
            })
    
    # Run async processing
    asyncio.run(process_all())
    
    # Calculate summary statistics
    if summary_stats['total_tested'] > 0:
        summary_stats['avg_permutations_generated'] = total_permutations / summary_stats['total_tested']
        summary_stats['improvement_rate'] = summary_stats['permutations_improved'] / summary_stats['total_tested']
        summary_stats['yolo_sequence_accuracy'] = summary_stats['yolo_original_correct'] / summary_stats['total_tested']
        summary_stats['best_sequence_accuracy'] = summary_stats['best_permutation_correct'] / summary_stats['total_tested']
    
    # Save results
    output_data = {
        'configuration': {
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'max_permutations': MAX_PERMUTATIONS
        },
        'failed': failed_images,  # Include failed images
        'summary': summary_stats,
        'results': results
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Failed images: {summary_stats['failed_images']}")
    if failed_images:
        print(f"  Failed: {', '.join(failed_images)}")
    print(f"Total images tested: {summary_stats['total_tested']}")
    print(f"Images with no disagreements: {summary_stats['no_disagreements']}")
    print(f"Average permutations per image: {summary_stats['avg_permutations_generated']:.1f}")
    print(f"Images improved by permutations: {summary_stats['permutations_improved']}")
    print(f"Improvement rate: {summary_stats.get('improvement_rate', 0):.2%}")
    print(f"\nSequence Accuracy:")
    print(f"  YOLO Original: {summary_stats.get('yolo_sequence_accuracy', 0):.2%}")
    print(f"  Best Permutation: {summary_stats.get('best_sequence_accuracy', 0):.2%}")
    print(f"\nResults saved to: {OUTPUT_FILE}")
    
    if summary_stats['improvement_cases']:
        print(f"\nTop 5 Improvements:")
        top_improvements = sorted(
            summary_stats['improvement_cases'],
            key=lambda x: x['improvement'],
            reverse=True
        )[:5]
        for case in top_improvements:
            print(f"  {case['filename']}: {case['yolo_accuracy']:.2%} -> {case['best_accuracy']:.2%}")
            print(f"    YOLO: {case['yolo_original']}")
            print(f"    Best: {case['best_permutation']}")

if __name__ == "__main__":
    test_with_permutations()
