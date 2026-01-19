# from PIL import Image
# from typing import List
# import asyncio
# from functools import partial

# def clamp(v, lo, hi):
#     return max(lo, min(hi, v))

# def pad_to_square(img, pad_color=(255, 255, 255), extra_margin=4):
#     w, h = img.size
#     side = max(w, h) + extra_margin
#     new = Image.new("RGB", (side, side), pad_color)
#     new.paste(img, ((side - w) // 2, (side - h) // 2))
#     return new

# def _get_crops_sync(original, seg_results, cfg) -> List[Image.Image]:
#     """
#     Internal synchronous implementation of the cropping logic.
#     """
#     W, H = original.size
#     crops = []

#     for r in seg_results:
#         if len(r.boxes) == 0:
#             continue

#         for box in r.boxes:
#             x1, y1, x2, y2 = [int(v.item()) for v in box.xyxy[0]]

#             # margins
#             lm = int(cfg.get("left_margin", 0))
#             rm = int(cfg.get("right_margin", 0))
#             tm = int(cfg.get("top_margin", 0))
#             bm = int(cfg.get("bottom_margin", 0))
#             auto = cfg.get("auto_expand", True)

#             if auto:
#                 width = x2 - x1
#                 extra = int(0.05 * width)
#                 x1 -= extra
#                 x2 += extra

#             x1 += lm
#             y1 += tm
#             x2 += rm
#             y2 += bm

#             x1 = clamp(x1, 0, W-1)
#             y1 = clamp(y1, 0, H-1)
#             x2 = clamp(x2, 0, W-1)
#             y2 = clamp(y2, 0, H-1)

#             if x2 <= x1 or y2 <= y1:
#                 continue

#             crop = original.crop((x1+220, y1, x2, y2))
#             crop = pad_to_square(crop)
#             crops.append(crop)

#     return crops

# async def get_crops_from_segmenter(original, seg_results, cfg) -> List[Image.Image]:
#     loop = asyncio.get_running_loop()
#     # Run the CPU-bound PIL operations in a thread pool
#     return await loop.run_in_executor(
#         None, 
#         partial(_get_crops_sync, original, seg_results, cfg)
#     )

# src/helpers/crop_utils.py
from PIL import Image, ImageFilter, ImageEnhance
from typing import Any, List
import asyncio
import numpy as np

import numpy as np
from PIL import Image

def trim_to_digits_strict(img: Image.Image,
                          bg_threshold=235,
                          min_col_ink=0.04):
    """
    Very aggressive horizontal trim to keep only digit area
    """
    gray = img.convert("L")
    arr = np.array(gray)

    h, w = arr.shape

    ink = arr < bg_threshold
    col_density = ink.sum(axis=0) / h

    valid = np.where(col_density > min_col_ink)[0]

    if len(valid) == 0:
        return img

    left = int(valid[0])
    right = int(valid[-1]) + 1

    return img.crop((left, 0, right, h))

def zoom_for_digit_classifier(img: Image.Image,
                              target_height=160):
    w, h = img.size
    scale = target_height / h
    return img.resize(
        (int(w * scale), target_height),
        Image.BICUBIC
    )

def upscale_to_min_height(img: Image.Image, min_height: int = 96) -> Image.Image:
    w, h = img.size

    if h >= min_height:
        return img

    scale = min_height / h
    new_w = int(w * scale)
    new_h = int(h * scale)

    return img.resize((new_w, new_h), Image.BICUBIC)

def trim_to_digit_region(img: Image.Image, threshold=240, min_ink_ratio=0.02):
    """
    Auto-crops image horizontally to retain only columns with digit ink.
    """
    gray = img.convert("L")
    arr = np.array(gray)

    h, w = arr.shape

    # Binary mask of "ink"
    ink = arr < threshold

    # Column-wise ink ratio
    col_ink_ratio = ink.sum(axis=0) / h

    # Find columns that actually contain digits
    valid_cols = np.where(col_ink_ratio > min_ink_ratio)[0]

    if len(valid_cols) == 0:
        return img  # fallback

    left = int(valid_cols[0])
    right = int(valid_cols[-1]) + 1

    return img.crop((left, 0, right, h))

def get_cropped_image(original_image: Image.Image, model: Any, results: Any) -> Image.Image:
    pad_color = 255  # white background for grayscale

    for i, r in enumerate(results):
        W, H = original_image.size

        if len(r.boxes) == 0:
            print("  No objects detected.")
            continue

        for j, box in enumerate(r.boxes):
            # Bounding box coords (xyxy)
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [
                int(max(0, min(coord, dim)))
                for coord, dim in zip(coords, [W, H, W, H])
            ]

            cls = model.names[int(box.cls[0])]
            conf = box.conf[0].item()
            print(f"  Detected: {cls}, Confidence: {conf:.2f}, BBox: {[x1, y1, x2, y2]}")

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                print("  Skipping degenerate bbox.")
                continue

            # -------------------------------
            # Improved dynamic cropping
            # -------------------------------
            box_width = x2 - x1
            box_height = y2 - y1

            # Trim left part relative to box width
            left_trim = int(0.25 * box_width)

            # Add vertical breathing space
            pad_y = int(0.30 * box_height)

            cx1 = min(W, x1 + left_trim)
            cy1 = max(0, y1 - pad_y)
            cx2 = x2
            cy2 = min(H, y2 + pad_y)

            cropped = original_image.crop((cx1, cy1, cx2, cy2))

            # 1. remove left labels & empty boxes
            cropped = trim_to_digits_strict(cropped)

            # 2. zoom aggressively (THIS creates your shown output)
            cropped = zoom_for_digit_classifier(cropped, target_height=160)

            return cropped

    return None

async def get_cropped_image_async(original_image: Image.Image, model: Any, results: Any) -> Image.Image:
    """
    Async wrapper for get_cropped_image — runs the synchronous function in an executor.
    Usage:
        crop = await get_cropped_image_async(pil_img, acc_model, seg_results)
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, get_cropped_image, original_image, model, results)
