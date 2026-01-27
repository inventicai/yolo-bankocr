"""
Test script to evaluate the accuracy of YOLO and Keras models
on the first 100 test images from All_data_png folder.
"""

import asyncio
import os
import time
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import json

from src.models.yolo_wrapper import YoloWrapper
from src.keras_classifier import KerasDigitClassifier
from src.helpers.image_io import open_image
from src.helpers.crop_utils import get_cropped_image_async
from src.postprocess.sequence_builder import (
    build_sequence, 
    build_top_n_sequences, 
    build_sequences_with_keras,
    build_threshold_sequences_top_n,
    calculate_digit_accuracy
)


class ModelAccuracyTester:
    """Test accuracy of YOLO segmenter, YOLO digit classifier, and Keras digit classifier."""
    
    def __init__(self, config_path: str = "configs/default.yaml"):
        """Initialize the tester with configuration."""
        self.config_path = Path(config_path)
        self.cfg = self._load_config()
        self.ground_truth = {}
        self.results = {
            "yolo_only": [],
            "keras_only": [],
            "combined": []
        }
        
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_ground_truth(self, ground_truth_file: Path) -> Dict[str, str]:
        """Load ground truth data from the generated_account.txt file."""
        ground_truth = {}
        
        print(f"[+] Loading ground truth from: {ground_truth_file}")
        
        with open(ground_truth_file, 'r') as f:
            lines = f.readlines()
        
        # Skip header lines and parse the data
        for line in lines:
            line = line.strip()
            if not line or line.startswith("Generated") or line.startswith("==") or \
               line.startswith("Template") or line.startswith("Generated on"):
                continue
            
            # Format: "test_form_0001.png: 833447588304"
            if ": " in line:
                filename, account_number = line.split(": ", 1)
                ground_truth[filename] = account_number.strip()
        
        print(f"[+] Loaded {len(ground_truth)} ground truth entries")
        return ground_truth
    
    async def initialize_models(self):
        """Initialize YOLO and Keras models."""
        print("[+] Initializing models...")
        
        cwd = os.getcwd()
        seg_path = os.path.join(cwd, 'src', self.cfg["models"]["segmenter_path"])
        digit_path = os.path.join(cwd, 'src', self.cfg["models"]["digit_model_path"])
        
        # Initialize YOLO models
        self.segmenter = YoloWrapper(seg_path)
        self.digit_yolo = YoloWrapper(digit_path)
        
        # Initialize Keras classifier
        keras_model_path = self.cfg.get("keras_model_path", "models/DigitClassifier.keras")
        self.keras_classifier = KerasDigitClassifier(keras_model_path)
        
        print("[+] Models initialized successfully")
    
    async def process_single_image(self, image_path: Path, ground_truth: str) -> Dict:
        """Process a single image and return results for all model combinations."""
        print(f"  Processing: {image_path.name}")
        
        result = {
            "image_name": image_path.name,
            "ground_truth": ground_truth,
            "yolo_prediction": None,
            "keras_top1_prediction": None,
            "yolo_correct": False,
            "keras_correct": False,
            "segmenter_success": False,
            "digit_detection_count": 0,
            "top2_sequences": [],  # Top-2 with 10% threshold
            "top3_sequences": [],  # Top-3 with 20% threshold
            "gt_found_in_top2": False,
            "gt_found_in_top3": False,
            "best_digit_accuracy_top2": 0.0,
            "best_digit_accuracy_top3": 0.0,
            "digit_alternatives_top2": [],
            "digit_alternatives_top3": [],
            "error": None
        }
        
        try:
            # Load image (with enhancements: brightness=1.2, sharpness=1.5, pdf_dpi=300)
            pil_img = await open_image(image_path)
            
            # 1. Segmentation (YOLO with internal preprocessing)
            seg_conf = self.cfg["thresholds"]["segmenter"]
            seg_res = await self.segmenter.predict(pil_img, conf=seg_conf)
            
            # 2. Crop the account number region (with enhancements: trim + zoom to 160px height)
            crop = await get_cropped_image_async(pil_img, self.segmenter, seg_res)
            
            if crop is None:
                result["error"] = "Segmentation failed - no crop found"
                return result
            
            result["segmenter_success"] = True
            
            # 3. Digit detection (YOLO with internal preprocessing)
            digit_conf = self.cfg["thresholds"]["digit"]
            digit_res = await self.digit_yolo.predict(crop, conf=digit_conf)
            
            if len(digit_res) == 0 or len(digit_res[0].boxes) == 0:
                result["error"] = "No digits detected"
                return result
            
            result["digit_detection_count"] = len(digit_res[0].boxes)
            
            # 4a. YOLO-only prediction (top-1)
            yolo_sequences = build_top_n_sequences(digit_res[0], n=1)
            yolo_top1 = yolo_sequences[0] if len(yolo_sequences) > 0 else ""
            result["yolo_prediction"] = yolo_top1
            result["yolo_correct"] = (yolo_top1 == ground_truth)
            
            # Calculate digit-level accuracy for YOLO
            result["yolo_digit_matches"] = sum(1 for a, b in zip(yolo_top1, ground_truth) if a == b)
            result["yolo_digit_accuracy"] = calculate_digit_accuracy(yolo_top1, ground_truth)
            
            # 4b. Keras-only prediction (top-1 from Keras)
            _, keras_top1, _ = build_sequences_with_keras(
                digit_res[0], crop, self.keras_classifier
            )
            result["keras_top1_prediction"] = keras_top1
            result["keras_correct"] = (keras_top1 == ground_truth)
            
            # Calculate digit-level accuracy for Keras
            result["keras_digit_matches"] = sum(1 for a, b in zip(keras_top1, ground_truth) if a == b)
            result["keras_digit_accuracy"] = calculate_digit_accuracy(keras_top1, ground_truth)
            
            # 4c. Top-2 sequences with 20% threshold
            top2_seqs_probs, digit_alts_top2, _ = build_threshold_sequences_top_n(
                digit_res[0], crop, self.keras_classifier, 
                top_n=2, 
                threshold_percent=20, 
                max_sequences=100
            )
            
            # Calculate digit accuracy for each top-2 sequence
            top2_with_accuracy = []
            for seq, prob in top2_seqs_probs:
                digit_acc = calculate_digit_accuracy(seq, ground_truth)
                top2_with_accuracy.append({
                    "sequence": seq,
                    "probability": prob,
                    "digit_accuracy": digit_acc,
                    "exact_match": (seq == ground_truth)
                })
            
            result["top2_sequences"] = top2_with_accuracy
            result["digit_alternatives_top2"] = digit_alts_top2
            result["gt_found_in_top2"] = any(item["exact_match"] for item in top2_with_accuracy)
            result["best_digit_accuracy_top2"] = max([item["digit_accuracy"] for item in top2_with_accuracy], default=0.0)
            
            # 4d. Top-3 sequences with 30% threshold
            top3_seqs_probs, digit_alts_top3, _ = build_threshold_sequences_top_n(
                digit_res[0], crop, self.keras_classifier, 
                top_n=3, 
                threshold_percent=30, 
                max_sequences=100
            )
            
            # Calculate digit accuracy for each top-3 sequence
            top3_with_accuracy = []
            for seq, prob in top3_seqs_probs:
                digit_acc = calculate_digit_accuracy(seq, ground_truth)
                top3_with_accuracy.append({
                    "sequence": seq,
                    "probability": prob,
                    "digit_accuracy": digit_acc,
                    "exact_match": (seq == ground_truth)
                })
            
            result["top3_sequences"] = top3_with_accuracy
            result["digit_alternatives_top3"] = digit_alts_top3
            result["gt_found_in_top3"] = any(item["exact_match"] for item in top3_with_accuracy)
            result["best_digit_accuracy_top3"] = max([item["digit_accuracy"] for item in top3_with_accuracy], default=0.0)
            
            print(f"    GT: {ground_truth}")
            print(f"    YOLO: {yolo_top1} {'✓' if result['yolo_correct'] else '✗'} (Digit: {result['yolo_digit_accuracy']:.1f}%)")
            print(f"    Keras: {keras_top1} {'✓' if result['keras_correct'] else '✗'} (Digit: {result['keras_digit_accuracy']:.1f}%)")
            print(f"    Top-2 (20% thresh): {len(top2_with_accuracy)} sequences, Best digit acc: {result['best_digit_accuracy_top2']:.1f}%")
            if len(top2_with_accuracy) > 0:
                print(f"      Best 3: {[item['sequence'] for item in top2_with_accuracy[:3]]}")
                if result['gt_found_in_top2']:
                    print(f"      ✓ GT found in top-2 sequences!")
            print(f"    Top-3 (30% thresh): {len(top3_with_accuracy)} sequences, Best digit acc: {result['best_digit_accuracy_top3']:.1f}%")
            if len(top3_with_accuracy) > 0:
                print(f"      Best 3: {[item['sequence'] for item in top3_with_accuracy[:3]]}")
                if result['gt_found_in_top3']:
                    print(f"      ✓ GT found in top-3 sequences!")
            
        except Exception as e:
            result["error"] = str(e)
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    async def test_accuracy(self, data_dir: Path, num_images: int = 100):
        """Test accuracy on the first N images."""
        print(f"\n{'='*60}")
        print(f"Testing Model Accuracy on First {num_images} Images")
        print(f"{'='*60}\n")
        
        # Load ground truth
        ground_truth_file = data_dir / "generated_account.txt"
        self.ground_truth = self.load_ground_truth(ground_truth_file)
        
        # Initialize models
        await self.initialize_models()
        
        # Get image files (sorted)
        image_files = sorted([
            p for p in data_dir.iterdir()
            if p.suffix.lower() == ".png" and p.name.startswith("test_form_")
        ])
        
        # Limit to first N images
        test_images = image_files[:num_images]
        print(f"[+] Testing on {len(test_images)} images\n")
        
        # Process each image
        start_time = time.time()
        all_results = []
        
        for i, img_path in enumerate(test_images, 1):
            print(f"[{i}/{len(test_images)}]")
            gt = self.ground_truth.get(img_path.name, "UNKNOWN")
            result = await self.process_single_image(img_path, gt)
            all_results.append(result)
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        stats = self._calculate_statistics(all_results)
        stats["total_processing_time"] = total_time
        stats["average_time_per_image"] = total_time / len(test_images)
        
        # Print summary
        self._print_summary(stats, len(test_images))
        
        # Save detailed results
        self._save_results(all_results, stats)
        
        return stats
    
    def _calculate_statistics(self, results: List[Dict]) -> Dict:
        """Calculate accuracy statistics from results."""
        stats = {
            "yolo": {
                "correct": 0,
                "incorrect": 0,
                "errors": 0,
                "accuracy": 0.0,
                "digit_accuracy": 0.0,
                "total_digit_matches": 0,
                "total_expected_digits": 0
            },
            "keras": {
                "correct": 0,
                "incorrect": 0,
                "errors": 0,
                "accuracy": 0.0,
                "digit_accuracy": 0.0,
                "total_digit_matches": 0,
                "total_expected_digits": 0
            },
            "threshold_sequences": {
                "top2_20_percent": {
                    "gt_found": 0,
                    "best_digit_accuracy_sum": 0.0,
                    "avg_best_digit_accuracy": 0.0,
                    "total_sequences": 0,
                    "avg_sequences_per_image": 0.0,
                    "recall": 0.0
                },
                "top3_30_percent": {
                    "gt_found": 0,
                    "best_digit_accuracy_sum": 0.0,
                    "avg_best_digit_accuracy": 0.0,
                    "total_sequences": 0,
                    "avg_sequences_per_image": 0.0,
                    "recall": 0.0
                }
            },
            "segmentation": {
                "success": 0,
                "failed": 0,
                "success_rate": 0.0
            },
            "digit_detection": {
                "total_digits": 0,
                "avg_digits_per_image": 0.0
            }
        }
        
        total_valid = 0
        
        for result in results:
            if result["error"]:
                stats["yolo"]["errors"] += 1
                stats["keras"]["errors"] += 1
                
                if not result["segmenter_success"]:
                    stats["segmentation"]["failed"] += 1
            else:
                total_valid += 1
                stats["segmentation"]["success"] += 1
                stats["digit_detection"]["total_digits"] += result["digit_detection_count"]
                
                # Track expected digits (12 per account number)
                expected_digits = len(result["ground_truth"])
                
                # YOLO stats
                if result["yolo_correct"]:
                    stats["yolo"]["correct"] += 1
                else:
                    stats["yolo"]["incorrect"] += 1
                stats["yolo"]["total_digit_matches"] += result.get("yolo_digit_matches", 0)
                stats["yolo"]["total_expected_digits"] += expected_digits
                
                # Keras stats
                if result["keras_correct"]:
                    stats["keras"]["correct"] += 1
                else:
                    stats["keras"]["incorrect"] += 1
                stats["keras"]["total_digit_matches"] += result.get("keras_digit_matches", 0)
                stats["keras"]["total_expected_digits"] += expected_digits
                
                # Top-2 threshold sequences stats
                top2_seqs = result.get("top2_sequences", [])
                if top2_seqs:
                    stats["threshold_sequences"]["top2_20_percent"]["total_sequences"] += len(top2_seqs)
                    if result.get("gt_found_in_top2", False):
                        stats["threshold_sequences"]["top2_20_percent"]["gt_found"] += 1
                    stats["threshold_sequences"]["top2_20_percent"]["best_digit_accuracy_sum"] += result.get("best_digit_accuracy_top2", 0.0)
                
                # Top-3 threshold sequences stats
                top3_seqs = result.get("top3_sequences", [])
                if top3_seqs:
                    stats["threshold_sequences"]["top3_30_percent"]["total_sequences"] += len(top3_seqs)
                    if result.get("gt_found_in_top3", False):
                        stats["threshold_sequences"]["top3_30_percent"]["gt_found"] += 1
                    stats["threshold_sequences"]["top3_30_percent"]["best_digit_accuracy_sum"] += result.get("best_digit_accuracy_top3", 0.0)
        
        # Calculate accuracies
        if total_valid > 0:
            stats["yolo"]["accuracy"] = (stats["yolo"]["correct"] / total_valid) * 100
            stats["keras"]["accuracy"] = (stats["keras"]["correct"] / total_valid) * 100
            stats["digit_detection"]["avg_digits_per_image"] = stats["digit_detection"]["total_digits"] / total_valid
            
            # Top-2 averages
            total_top2 = stats["threshold_sequences"]["top2_20_percent"]["total_sequences"]
            if total_top2 > 0:
                stats["threshold_sequences"]["top2_20_percent"]["avg_sequences_per_image"] = total_top2 / total_valid
            stats["threshold_sequences"]["top2_20_percent"]["recall"] = (
                stats["threshold_sequences"]["top2_20_percent"]["gt_found"] / total_valid
            ) * 100
            stats["threshold_sequences"]["top2_20_percent"]["avg_best_digit_accuracy"] = (
                stats["threshold_sequences"]["top2_20_percent"]["best_digit_accuracy_sum"] / total_valid
            )
            
            # Top-3 averages
            total_top3 = stats["threshold_sequences"]["top3_30_percent"]["total_sequences"]
            if total_top3 > 0:
                stats["threshold_sequences"]["top3_30_percent"]["avg_sequences_per_image"] = total_top3 / total_valid
            stats["threshold_sequences"]["top3_30_percent"]["recall"] = (
                stats["threshold_sequences"]["top3_30_percent"]["gt_found"] / total_valid
            ) * 100
            stats["threshold_sequences"]["top3_30_percent"]["avg_best_digit_accuracy"] = (
                stats["threshold_sequences"]["top3_30_percent"]["best_digit_accuracy_sum"] / total_valid
            )
        
        # Calculate digit-level accuracies
        if stats["yolo"]["total_expected_digits"] > 0:
            stats["yolo"]["digit_accuracy"] = (stats["yolo"]["total_digit_matches"] / stats["yolo"]["total_expected_digits"]) * 100
        if stats["keras"]["total_expected_digits"] > 0:
            stats["keras"]["digit_accuracy"] = (stats["keras"]["total_digit_matches"] / stats["keras"]["total_expected_digits"]) * 100
        
        if len(results) > 0:
            stats["segmentation"]["success_rate"] = (stats["segmentation"]["success"] / len(results)) * 100
        
        return stats
    
    def _print_summary(self, stats: Dict, total_images: int):
        """Print summary of test results."""
        print(f"\n{'='*60}")
        print("ACCURACY TEST SUMMARY")
        print(f"{'='*60}\n")
        
        print(f"Total Images Tested: {total_images}")
        print(f"Processing Time: {stats['total_processing_time']:.2f}s")
        print(f"Avg Time per Image: {stats['average_time_per_image']:.3f}s\n")
        
        total_valid = stats['segmentation']['success']
        
        print(f"{'SEGMENTATION (YOLO)':-^60}")
        print(f"  Success: {stats['segmentation']['success']} / {total_images}")
        print(f"  Success Rate: {stats['segmentation']['success_rate']:.2f}%\n")
        
        print(f"{'MODEL PREDICTIONS':-^60}")
        
        # YOLO Only
        print(f"\n  1. YOLO Prediction (Detection + Classification):")
        print(f"     Correct Sequences:   {stats['yolo']['correct']} / {total_valid}")
        print(f"     Sequence Accuracy:   {stats['yolo']['accuracy']:.2f}%")
        print(f"     Digit-Level Accuracy: {stats['yolo']['digit_accuracy']:.2f}%")
        
        # Keras Only
        print(f"\n  2. Keras Prediction (Top-1 from Keras):")
        print(f"     Correct Sequences:   {stats['keras']['correct']} / {total_valid}")
        print(f"     Sequence Accuracy:   {stats['keras']['accuracy']:.2f}%")
        print(f"     Digit-Level Accuracy: {stats['keras']['digit_accuracy']:.2f}%")
        
        print(f"\n{'THRESHOLD-BASED SEQUENCES (Top-N from Keras)':-^60}")
        
        # Top-2 with 20% threshold
        top2_stats = stats['threshold_sequences']['top2_20_percent']
        print(f"\n  3. Top-2 Sequences (20% threshold):")
        print(f"     Total Sequences Generated: {top2_stats['total_sequences']}")
        print(f"     Avg Sequences per Image: {top2_stats['avg_sequences_per_image']:.1f}")
        print(f"     GT Found (Recall): {top2_stats['gt_found']} / {total_valid} ({top2_stats['recall']:.2f}%)")
        print(f"     Avg Best Digit Accuracy: {top2_stats['avg_best_digit_accuracy']:.2f}%")
        
        # Top-3 with 30% threshold
        top3_stats = stats['threshold_sequences']['top3_30_percent']
        print(f"\n  4. Top-3 Sequences (30% threshold):")
        print(f"     Total Sequences Generated: {top3_stats['total_sequences']}")
        print(f"     Avg Sequences per Image: {top3_stats['avg_sequences_per_image']:.1f}")
        print(f"     GT Found (Recall): {top3_stats['gt_found']} / {total_valid} ({top3_stats['recall']:.2f}%)")
        print(f"     Avg Best Digit Accuracy: {top3_stats['avg_best_digit_accuracy']:.2f}%")
        
        print(f"\n{'IMAGE ENHANCEMENTS APPLIED':-^60}")
        print(f"  Image Loading: Brightness=1.2, Sharpness=1.5, PDF_DPI=300")
        
        print(f"\n{'IMAGE ENHANCEMENTS APPLIED':-^60}")
        print(f"  YOLO Segmenter: Internal YOLO preprocessing")
        print(f"  YOLO Digit Detector: Internal YOLO preprocessing")
        print(f"  Crop Processing: Trimming + Zooming (target height: 160px)")
        print(f"  Keras Classifier: Grayscale + Resize to 28x28 + Normalization")
        
        print(f"\n{'='*60}\n")
    
    def _save_results(self, results: List[Dict], stats: Dict):
        """Save detailed results to JSON file."""
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "model_accuracy_test_results.json"
        
        output_data = {
            "test_config": {
                "num_images": len(results),
                "config_file": str(self.config_path),
                "data_directory": "All_data_png"
            },
            "statistics": stats,
            "detailed_results": results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"[+] Detailed results saved to: {output_file}")


async def main():
    """Main function to run the accuracy test."""
    # Configuration
    data_dir = Path("All_data_png")
    config_path = "configs/default.yaml"
    num_images = 100
    
    # Verify data directory exists
    if not data_dir.exists():
        print(f"[!] Error: Data directory not found: {data_dir}")
        return
    
    # Create tester and run tests
    tester = ModelAccuracyTester(config_path=config_path)
    
    try:
        stats = await tester.test_accuracy(data_dir, num_images=num_images)
        print("\n[✓] Accuracy testing completed successfully!")
        
    except Exception as e:
        print(f"\n[!] Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
