"""
Test script to calculate model accuracy on first 100 images from All_data_png.
Calculates both sequence-level accuracy and digit-level accuracy.
Compares YOLO classification vs Keras classification on the same bounding boxes.
"""
import asyncio
import os
import time
import yaml
from pathlib import Path
from src.models.yolo_wrapper import YoloWrapper
from src.models.keras_wrapper import KerasDigitClassifier
from src.helpers.image_io import open_image
from src.helpers.crop_utils import get_cropped_image_async
from src.postprocess.sequence_builder import build_sequence
from src.postprocess.keras_sequence_builder import build_sequence_with_keras


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


def calculate_digit_accuracy(predicted: str, actual: str):
    """Calculate digit-level accuracy between predicted and actual sequences."""
    if not predicted and not actual:
        return 1.0, 0, 0
    
    # Pad shorter sequence with empty spaces for comparison
    max_len = max(len(predicted), len(actual))
    pred_padded = predicted.ljust(max_len, ' ')
    act_padded = actual.ljust(max_len, ' ')
    
    correct_digits = sum(1 for p, a in zip(pred_padded, act_padded) if p == a and a != ' ')
    total_digits = len(actual)  # Use actual length as denominator
    
    accuracy = correct_digits / total_digits if total_digits > 0 else 0.0
    return accuracy, correct_digits, total_digits


async def test_accuracy(num_images=100):
    """Test model accuracy on first N images."""
    CWD = os.getcwd()
    
    # Load config
    config_path = Path("configs/default.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    cfg = yaml.safe_load(open(config_path, 'r'))
    
    # Load models
    seg_path = os.path.join(CWD, 'src', cfg["models"]["segmenter_path"])
    digit_path = os.path.join(CWD, 'src', cfg["models"]["digit_model_path"])
    keras_path = os.path.join(CWD, 'src', 'models', 'DigitClassifier.keras')
    seg_conf = cfg["thresholds"]["segmenter"]
    digit_conf = cfg["thresholds"]["digit"]
    
    print("[+] Loading models...")
    seg = YoloWrapper(seg_path)
    digit = YoloWrapper(digit_path)
    
    # Load Keras model
    try:
        keras_classifier = KerasDigitClassifier(keras_path)
        use_keras = True
    except Exception as e:
        print(f"[!] Warning: Could not load Keras model: {e}")
        print("[!] Continuing with YOLO-only predictions...")
        use_keras = False
        keras_classifier = None
    
    print("[+] Models loaded successfully!\n")
    
    # Load ground truth
    data_dir = Path("All_data_png")
    gt_file = data_dir / "generated_account.txt"
    
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
    
    print("[+] Loading ground truth...")
    ground_truth = load_ground_truth(gt_file)
    print(f"[+] Loaded {len(ground_truth)} ground truth entries\n")
    
    # Get first N images
    image_files = sorted([
        p for p in data_dir.iterdir()
        if p.suffix.lower() == '.png' and p.name.startswith('test_form_')
    ])[:num_images]
    
    print(f"[+] Testing on {len(image_files)} images\n")
    print("=" * 80)
    
    # Initialize counters
    total_tested = 0
    sequence_correct_yolo = 0
    sequence_correct_keras = 0
    total_digits = 0
    correct_digits_yolo = 0
    correct_digits_keras = 0
    failed_images = []
    results = []
    
    start_time = time.time()
    
    # Process each image
    for i, img_path in enumerate(image_files, 1):
        filename = img_path.name
        
        # Get ground truth
        if filename not in ground_truth:
            print(f"[{i}/{len(image_files)}] {filename}: No ground truth found, skipping...")
            continue
        
        actual = ground_truth[filename]
        
        try:
            # Load image
            pil_img = await open_image(img_path)
            
            # Run segmenter
            seg_res = await seg.predict(pil_img, conf=seg_conf)
            
            # Crop
            crop = await get_cropped_image_async(pil_img, seg, seg_res)
            if crop is None:
                print(f"[{i}/{len(image_files)}] {filename}: No crop found")
                failed_images.append((filename, "No crop"))
                continue
            
            # Run digit classifier
            digit_res = await digit.predict(crop, conf=digit_conf)
            
            # Extract sequence using YOLO classification
            predicted_yolo = build_sequence(digit_res[0]) if len(digit_res) > 0 else ""
            
            # Extract sequence using Keras classification
            predicted_keras = ""
            if use_keras and len(digit_res) > 0:
                predicted_keras = await build_sequence_with_keras(crop, digit_res[0], keras_classifier)
            
            # Calculate accuracy for both methods
            digit_acc_yolo, correct_d_yolo, total_d = calculate_digit_accuracy(predicted_yolo, actual)
            digit_acc_keras, correct_d_keras, _ = calculate_digit_accuracy(predicted_keras, actual) if use_keras else (0.0, 0, 0)
            
            # Update counters
            total_tested += 1
            correct_digits_yolo += correct_d_yolo
            correct_digits_keras += correct_d_keras
            total_digits += total_d
            
            # Check if sequence is completely correct
            is_correct_yolo = (predicted_yolo == actual)
            is_correct_keras = (predicted_keras == actual) if use_keras else False
            if is_correct_yolo:
                sequence_correct_yolo += 1
            if is_correct_keras:
                sequence_correct_keras += 1
            
            # Store result
            results.append({
                'filename': filename,
                'actual': actual,
                'predicted_yolo': predicted_yolo,
                'predicted_keras': predicted_keras,
                'digit_accuracy_yolo': digit_acc_yolo,
                'digit_accuracy_keras': digit_acc_keras,
                'sequence_correct_yolo': is_correct_yolo,
                'sequence_correct_keras': is_correct_keras
            })
            
            # Print progress
            status_yolo = "✓" if is_correct_yolo else "✗"
            status_keras = "✓" if is_correct_keras else "✗" if use_keras else "-"
            print(f"[{i}/{len(image_files)}] {filename}")
            print(f"    Actual:          {actual}")
            print(f"    YOLO {status_yolo}:         {predicted_yolo} (Accuracy: {digit_acc_yolo*100:.1f}% - {correct_d_yolo}/{total_d})")
            if use_keras:
                print(f"    Keras {status_keras}:        {predicted_keras} (Accuracy: {digit_acc_keras*100:.1f}% - {correct_d_keras}/{total_d})")
            print()
            
        except Exception as e:
            print(f"[{i}/{len(image_files)}] {filename}: Error - {e}")
            failed_images.append((filename, str(e)))
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate final metrics
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total images tested: {total_tested}")
    print(f"Failed images: {len(failed_images)}")
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Average time per image: {total_time/len(image_files):.2f}s")
    print()
    
    if total_tested > 0:
        seq_accuracy_yolo = (sequence_correct_yolo / total_tested) * 100
        digit_accuracy_yolo = (correct_digits_yolo / total_digits) * 100 if total_digits > 0 else 0
        
        print("=" * 80)
        print("YOLO MODEL RESULTS")
        print("=" * 80)
        print(f"SEQUENCE-LEVEL ACCURACY: {seq_accuracy_yolo:.2f}% ({sequence_correct_yolo}/{total_tested})")
        print(f"DIGIT-LEVEL ACCURACY:    {digit_accuracy_yolo:.2f}% ({correct_digits_yolo}/{total_digits})")
        print()
        
        if use_keras:
            seq_accuracy_keras = (sequence_correct_keras / total_tested) * 100
            digit_accuracy_keras = (correct_digits_keras / total_digits) * 100 if total_digits > 0 else 0
            
            print("=" * 80)
            print("KERAS MODEL RESULTS")
            print("=" * 80)
            print(f"SEQUENCE-LEVEL ACCURACY: {seq_accuracy_keras:.2f}% ({sequence_correct_keras}/{total_tested})")
            print(f"DIGIT-LEVEL ACCURACY:    {digit_accuracy_keras:.2f}% ({correct_digits_keras}/{total_digits})")
            print()
            
            # Comparison
            print("=" * 80)
            print("COMPARISON")
            print("=" * 80)
            diff_seq = seq_accuracy_keras - seq_accuracy_yolo
            diff_digit = digit_accuracy_keras - digit_accuracy_yolo
            print(f"Sequence Accuracy Difference: {diff_seq:+.2f}% (Keras vs YOLO)")
            print(f"Digit Accuracy Difference:    {diff_digit:+.2f}% (Keras vs YOLO)")
            print()
    
    # Show failed images
    if failed_images:
        print("\n" + "=" * 80)
        print("FAILED IMAGES")
        print("=" * 80)
        for filename, error in failed_images:
            print(f"  - {filename}: {error}")
    
    # Show worst performing images (only errors) for YOLO
    if results:
        print("\n" + "=" * 80)
        print("YOLO: IMAGES WITH ERRORS (sorted by digit accuracy)")
        print("=" * 80)
        
        error_results = [r for r in results if not r['sequence_correct_yolo']]
        error_results.sort(key=lambda x: x['digit_accuracy_yolo'])
        
        for r in error_results[:10]:  # Show top 10 errors
            print(f"\n{r['filename']} - Digit Accuracy: {r['digit_accuracy_yolo']*100:.1f}%")
            print(f"  Actual:    {r['actual']}")
            print(f"  Predicted: {r['predicted_yolo']}")
        
        if use_keras:
            print("\n" + "=" * 80)
            print("KERAS: IMAGES WITH ERRORS (sorted by digit accuracy)")
            print("=" * 80)
            
            error_results_keras = [r for r in results if not r['sequence_correct_keras']]
            error_results_keras.sort(key=lambda x: x['digit_accuracy_keras'])
            
            for r in error_results_keras[:10]:  # Show top 10 errors
                print(f"\n{r['filename']} - Digit Accuracy: {r['digit_accuracy_keras']*100:.1f}%")
                print(f"  Actual:    {r['actual']}")
                print(f"  Predicted: {r['predicted_keras']}")
    
    # Save detailed results to JSON
    import json
    output_file = "test_accuracy_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_tested': total_tested,
                'yolo': {
                    'sequence_accuracy': seq_accuracy_yolo if total_tested > 0 else 0,
                    'digit_accuracy': digit_accuracy_yolo if total_digits > 0 else 0,
                    'sequence_correct': sequence_correct_yolo,
                    'correct_digits': correct_digits_yolo,
                },
                'keras': {
                    'sequence_accuracy': seq_accuracy_keras if (use_keras and total_tested > 0) else 0,
                    'digit_accuracy': digit_accuracy_keras if (use_keras and total_digits > 0) else 0,
                    'sequence_correct': sequence_correct_keras if use_keras else 0,
                    'correct_digits': correct_digits_keras if use_keras else 0,
                } if use_keras else None,
                'total_digits': total_digits,
                'failed_images': len(failed_images),
                'total_time': total_time
            },
            'results': results,
            'failed': [{'filename': f, 'error': e} for f, e in failed_images]
        }, f, indent=2)
    
    print(f"\n[+] Detailed results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test YOLO OCR model accuracy")
    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of images to test (default: 100)"
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("YOLO OCR Accuracy Test")
    print("=" * 80)
    print()
    
    asyncio.run(test_accuracy(args.num_images))
