from pathlib import Path
import json
import os
import time
import asyncio
from functools import partial
from src.models.yolo_wrapper import YoloWrapper
from src.keras_classifier import KerasDigitClassifier
from src.helpers.image_io import open_image
from src.helpers.crop_utils import get_cropped_image_async
from src.postprocess.sequence_builder import build_sequence, build_top_n_sequences, build_sequences_with_keras
from src.helpers.visualizer import draw_segmenter_boxes

from io import BytesIO
import fitz  # PyMuPDF for PDF processing
from PIL import Image
import numpy as np

CWD = os.getcwd()

async def run_folder(input_dir: Path, cfg: dict):
    seg_path = os.path.join(CWD, 'src', cfg["models"]["segmenter_path"])
    digit_path = os.path.join(CWD, 'src', cfg["models"]["digit_model_path"])
    seg_conf = cfg["thresholds"]["segmenter"]
    digit_conf = cfg["thresholds"]["digit"]
    out_json = cfg["output"]["json"]
    use_keras = cfg.get("use_keras_classifier", False)

    seg = YoloWrapper(seg_path)
    digit = YoloWrapper(digit_path)
    
    # Initialize Keras classifier if enabled
    keras_classifier = None
    if use_keras:
        keras_model_path = cfg.get("keras_model_path", "hf://lizardwine/DigitClassifier")
        keras_classifier = KerasDigitClassifier(keras_model_path)

    results_list = []

    files = [
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".pdf"]
    ]
    files = sorted(files)

    # ---- GLOBAL TIMER START ----
    global_start = time.time()

    for p in files:
        print(f"[+] Processing {p.name}")
        
        # ---- IMAGE TIMER START ----
        image_start = time.time()

        # Load image (async)
        try:
            pil_img = await open_image(p)
        except Exception as e:
            print(f"  ! Failed to open {p.name}: {e}")
            continue

        # 1. Segmenter
        t0 = time.time()
        seg_res = await seg.predict(pil_img, conf=seg_conf)
        seg_time = time.time() - t0

        # Segmenter visualization
        vis_img = draw_segmenter_boxes(pil_img, seg_res)
        vis_dir = Path("outputs/segmenter_vis")
        vis_dir.mkdir(parents=True, exist_ok=True)
        vis_path = vis_dir / f"{p.stem}_segmenter.png"
        vis_img.save(vis_path)
        print(f"  -> Saved segmenter visualization: {vis_path}")

        # 2. Crop
        crop = await get_cropped_image_async(pil_img, seg, seg_res)
        if crop is None:
            print("  - No crops found.")
            continue

        out_crop_dir = Path("outputs/crops")
        out_crop_dir.mkdir(parents=True, exist_ok=True)
        crop_path = out_crop_dir / f"{p.stem}_crop.png"

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, crop.save, str(crop_path))
        print(f"  -> Saved crop image: {crop_path}")

        # 3. Digit detection/classification
        t1 = time.time()
        digit_res = await digit.predict(crop, conf=digit_conf)
        digit_time = time.time() - t1

        # 4. Extract sequences (top-1 and top-2)
        probabilities = {}
        if len(digit_res) > 0:
            if use_keras and keras_classifier:
                # Top-1 from YOLO, Top-2 from Keras (with batch inference)
                seq_top1, seq_top2, probabilities = build_sequences_with_keras(digit_res[0], crop, keras_classifier)
                print(f"  -> Top-1 Sequence (YOLO): {seq_top1}")
                print(f"  -> Top-2 Sequence (Keras): {seq_top2}")
            else:
                # Use YOLO only for both
                sequences = build_top_n_sequences(digit_res[0], n=2)
                seq_top1 = sequences[0]
                seq_top2 = sequences[1]
                print(f"  -> Top-1 Sequence (YOLO): {seq_top1}")
                print(f"  -> Top-2 Sequence (YOLO): {seq_top2}")
        else:
            seq_top1 = ""
            seq_top2 = ""

        # ---- IMAGE TIMER END ----
        image_end = time.time()
        total_image_time = image_end - image_start

        print(f"  -> Total time for {p.name}: {total_image_time:.3f}s\n")

        # Store results
        result_data = {
            "image_name": p.name,
            "account_number": seq_top1,
            "account_number_top2": seq_top2,
            "timings": {
                "segmenter": seg_time,
                "digit": digit_time,
                "total_image_time": total_image_time
            }
        }
        if probabilities:
            result_data["probabilities"] = probabilities
        results_list.append(result_data)
        
    # ---- GLOBAL TIMER END ----
    global_end = time.time()
    global_total_time = global_end - global_start

    print("\n===============================")
    print(f"Total inference time: {global_total_time:.3f}s")
    print("===============================\n")

    # Save JSON
    os.makedirs(Path(out_json).parent, exist_ok=True)
    def _write_json():
        with open(out_json, "w") as f:
            json.dump({
                "results": results_list,
                "global_total_time": global_total_time
            }, f, indent=2)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _write_json)
    print(f"[+] Saved predictions → {out_json}")
    
async def process_pdf_on_the_fly(pdf_file: bytes, cfg: dict, pdf_name: str = None):
    """
    Process a PDF file directly from bytes without storing it permanently.
    
    Args:
        pdf_file: PDF file bytes or BytesIO object
        cfg: Configuration dictionary
        pdf_name: Optional name for the PDF file (for output naming)
    
    Returns:
        Dictionary with results for each page
    """
    seg_path = os.path.join(CWD, 'src', cfg["models"]["segmenter_path"])
    digit_path = os.path.join(CWD, 'src', cfg["models"]["digit_model_path"])
    seg_conf = cfg["thresholds"]["segmenter"]
    digit_conf = cfg["thresholds"]["digit"]
    out_json = cfg["output"]["json"]
    use_keras = cfg.get("use_keras_classifier", False)
    
    seg = YoloWrapper(seg_path)
    digit = YoloWrapper(digit_path)
    
    # Initialize Keras classifier if enabled
    keras_classifier = None
    if use_keras:
        keras_model_path = cfg.get("keras_model_path", "hf://lizardwine/DigitClassifier")
        keras_classifier = KerasDigitClassifier(keras_model_path)
    
    results_list = []
    
    # ---- GLOBAL TIMER START ----
    global_start = time.time()
    
    # Create temporary file or use BytesIO
    if isinstance(pdf_file, bytes):
        pdf_bytes_io = BytesIO(pdf_file)
    else:
        pdf_bytes_io = pdf_file
    
    try:
        # Open PDF from bytes
        doc = fitz.open(stream=pdf_bytes_io, filetype="pdf")
        num_pages = len(doc)
        print(f"[+] Processing PDF with {num_pages} pages")
        
        for page_num in range(num_pages):
            print(f"\n[+] Processing page {page_num + 1}/{num_pages}")
            
            # ---- PAGE TIMER START ----
            page_start = time.time()
            
            page = doc[page_num]
            
            # Convert PDF page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
            img_data = pix.tobytes("ppm")
            
            # Convert to PIL Image
            pil_img = Image.open(BytesIO(img_data))
            
            # Generate a name for this page
            if pdf_name:
                page_name = f"{Path(pdf_name).stem}_page_{page_num + 1}"
            else:
                page_name = f"pdf_page_{page_num + 1}"
            
            # 1. Segmenter
            t0 = time.time()
            seg_res = await seg.predict(pil_img, conf=seg_conf)
            seg_time = time.time() - t0
            
            # Segmenter visualization
            vis_img = draw_segmenter_boxes(pil_img, seg_res)
            vis_dir = Path("outputs/segmenter_vis")
            vis_dir.mkdir(parents=True, exist_ok=True)
            vis_path = vis_dir / f"{page_name}_segmenter.png"
            vis_img.save(vis_path)
            print(f"  -> Saved segmenter visualization: {vis_path}")
            
            # 2. Crop
            crop = await get_cropped_image_async(pil_img, seg, seg_res)
            if crop is None:
                print("  - No crops found.")
                # Still record the page with empty results
                results_list.append({
                    "page": page_num + 1,
                    "image_name": f"{page_name}.png",
                    "account_number": "",
                    "account_number_top2": "",
                    "timings": {
                        "segmenter": seg_time,
                        "digit": 0.0,
                        "total_page_time": time.time() - page_start
                    }
                })
                continue
            
            # Save cropped image
            out_crop_dir = Path("outputs/crops")
            out_crop_dir.mkdir(parents=True, exist_ok=True)
            crop_path = out_crop_dir / f"{page_name}_crop.png"
            
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, crop.save, str(crop_path))
            print(f"  -> Saved crop image: {crop_path}")
            
            # 3. Digit detection/classification
            t1 = time.time()
            digit_res = await digit.predict(crop, conf=digit_conf)
            digit_time = time.time() - t1
            
            # 4. Extract sequences (top-1 and top-2)
            probabilities = {}
            if len(digit_res) > 0:
                if use_keras and keras_classifier:
                    # Top-1 from YOLO, Top-2 from Keras (with batch inference)
                    seq_top1, seq_top2, probabilities = build_sequences_with_keras(digit_res[0], crop, keras_classifier)
                    print(f"  -> Top-1 Sequence (YOLO): {seq_top1}")
                    print(f"  -> Top-2 Sequence (Keras): {seq_top2}")
                else:
                    # Use YOLO only for both
                    sequences = build_top_n_sequences(digit_res[0], n=2)
                    seq_top1 = sequences[0]
                    seq_top2 = sequences[1]
                    print(f"  -> Top-1 Sequence (YOLO): {seq_top1}")
                    print(f"  -> Top-2 Sequence (YOLO): {seq_top2}")
            else:
                seq_top1 = ""
                seq_top2 = ""
            
            # ---- PAGE TIMER END ----
            page_end = time.time()
            total_page_time = page_end - page_start
            
            print(f"  -> Total time for page {page_num + 1}: {total_page_time:.3f}s")
            
            # Store results
            result_data = {
                "page": page_num + 1,
                "image_name": f"{page_name}.png",
                "account_number": seq_top1,
                "account_number_top2": seq_top2,
                "timings": {
                    "segmenter": seg_time,
                    "digit": digit_time,
                    "total_page_time": total_page_time
                }
            }
            if probabilities:
                result_data["probabilities"] = probabilities
            results_list.append(result_data)
        
        doc.close()
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise
    
    # ---- GLOBAL TIMER END ----
    global_end = time.time()
    global_total_time = global_end - global_start
    
    print("\n" + "=" * 40)
    print(f"Total processing time: {global_total_time:.3f}s")
    print(f"Processed {len(results_list)} pages")
    print("=" * 40 + "\n")
    
    # Prepare final result
    result_data = {
        "pdf_name": pdf_name or "uploaded_pdf",
        "total_pages": num_pages,
        "results": results_list,
        "global_total_time": global_total_time
    }
    
    # Save JSON if output path is specified
    if out_json:
        os.makedirs(Path(out_json).parent, exist_ok=True)
        def _write_json():
            with open(out_json, "w") as f:
                json.dump(result_data, f, indent=2)
        
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _write_json)
        print(f"[+] Saved predictions → {out_json}")
    
    return result_data


# For API usage with FastAPI/Flask:
async def process_pdf_api_endpoint(file_upload, config: dict):
    """
    Example API endpoint function for processing uploaded PDFs
    """
    try:
        # Read uploaded file
        pdf_bytes = await file_upload.read()
        
        # Process PDF
        results = await process_pdf_on_the_fly(
            pdf_file=pdf_bytes,
            cfg=config,
            pdf_name=file_upload.filename
        )
        
        return {
            "success": True,
            "data": results
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# Alternative: Process single image file on-the-fly
async def process_image_on_the_fly(image_file: bytes, cfg: dict, image_name: str = None):
    """
    Process a single image file directly from bytes
    """
    seg_path = os.path.join(CWD, 'src', cfg["models"]["segmenter_path"])
    digit_path = os.path.join(CWD, 'src', cfg["models"]["digit_model_path"])
    seg_conf = cfg["thresholds"]["segmenter"]
    digit_conf = cfg["thresholds"]["digit"]
    out_json = cfg["output"]["json"]
    use_keras = cfg.get("use_keras_classifier", False)
    
    seg = YoloWrapper(seg_path)
    digit = YoloWrapper(digit_path)
    
    # Initialize Keras classifier if enabled
    keras_classifier = None
    if use_keras:
        keras_model_path = cfg.get("keras_model_path", "hf://lizardwine/DigitClassifier")
        keras_classifier = KerasDigitClassifier(keras_model_path)
    
    global_start = time.time()
    
    try:
        # Convert bytes to PIL Image (without enhancements, like test_accuracy)
        pil_img = Image.open(BytesIO(image_file))
        
        # Convert to RGB if needed
        if pil_img.mode not in ("RGB", "RGBA"):
            pil_img = pil_img.convert("RGB")
        
        if image_name:
            image_stem = Path(image_name).stem
        else:
            image_stem = "uploaded_image"
        
        print(f"[+] Processing {image_stem}")
        image_start = time.time()
        
        # 1. Segmenter
        t0 = time.time()
        seg_res = await seg.predict(pil_img, conf=seg_conf)
        seg_time = time.time() - t0
        
        # Segmenter visualization
        vis_img = draw_segmenter_boxes(pil_img, seg_res)
        vis_dir = Path("outputs/segmenter_vis")
        vis_dir.mkdir(parents=True, exist_ok=True)
        vis_path = vis_dir / f"{image_stem}_segmenter.png"
        vis_img.save(vis_path)
        print(f"  -> Saved segmenter visualization: {vis_path}")
        
        # 2. Crop
        crop = await get_cropped_image_async(pil_img, seg, seg_res)
        if crop is None:
            print("  - No crops found.")
            return None
        
        out_crop_dir = Path("outputs/crops")
        out_crop_dir.mkdir(parents=True, exist_ok=True)
        crop_path = out_crop_dir / f"{image_stem}_crop.png"
        
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, crop.save, str(crop_path))
        print(f"  -> Saved crop image: {crop_path}")
        
        # 3. Digit detection/classification
        t1 = time.time()
        digit_res = await digit.predict(crop, conf=digit_conf)
        digit_time = time.time() - t1
        
        # 4. Extract sequences (top-1 and top-2)
        probabilities = {}
        if len(digit_res) > 0:
            if use_keras and keras_classifier:
                # Top-1 from YOLO, Top-2 from Keras (with batch inference)
                seq_top1, seq_top2, probabilities = build_sequences_with_keras(digit_res[0], crop, keras_classifier)
                print(f"  -> Top-1 Sequence (YOLO): {seq_top1}")
                print(f"  -> Top-2 Sequence (Keras): {seq_top2}")
            else:
                # Use YOLO only for both
                sequences = build_top_n_sequences(digit_res[0], n=2)
                seq_top1 = sequences[0]
                seq_top2 = sequences[1]
                print(f"  -> Top-1 Sequence (YOLO): {seq_top1}")
                print(f"  -> Top-2 Sequence (YOLO): {seq_top2}")
        else:
            seq_top1 = ""
            seq_top2 = ""
        
        image_end = time.time()
        total_image_time = image_end - image_start
        
        print(f"  -> Total processing time: {total_image_time:.3f}s")
        
        result = {
            "image_name": image_name or "uploaded_image",
            "account_number": seq_top1,
            "account_number_top2": seq_top2,
            "timings": {
                "segmenter": seg_time,
                "digit": digit_time,
                "total_image_time": total_image_time
            }
        }
        if probabilities:
            result["probabilities"] = probabilities
        
        global_end = time.time()
        
        # Save JSON if needed
        if out_json:
            os.makedirs(Path(out_json).parent, exist_ok=True)
            def _write_json():
                with open(out_json, "w") as f:
                    json.dump({
                        "results": [result],
                        "global_total_time": global_end - global_start
                    }, f, indent=2)
            
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _write_json)
            print(f"[+] Saved predictions → {out_json}")
        
        return result
        
    except Exception as e:
        print(f"Error processing image: {e}")
        raise
