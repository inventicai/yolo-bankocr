# src/helpers/image_io.py
from PIL import Image, ImageEnhance
import fitz  # PyMuPDF
from pathlib import Path
import asyncio
from functools import partial
from typing import Optional

# ===========================
# Internal synchronous worker
# ===========================
def _open_and_enhance_sync(
    path_like,
    brightness: float = 1.2,
    sharpness: float = 1.5,
    pdf_dpi: Optional[int] = 300,
):
    """
    Loads the image (or first page of PDF) synchronously, preserves/controls resolution for PDFs,
    enhances brightness & sharpness, and returns a PIL.Image object.

    Args:
        path_like: str or Path to image/pdf
        brightness: brightness multiplier (1.0 = no change)
        sharpness: sharpness multiplier (1.0 = no change)
        pdf_dpi: if int -> render PDF page at this DPI; if None -> use Matrix(1,1) (native vector scale)
                 default 300 for good OCR-quality images.
    """
    p = Path(path_like)
    suffix = p.suffix.lower()

    # ---- Load image or PDF ----
    if suffix == ".pdf":
        # Open document
        doc = fitz.open(str(p))
        page = doc.load_page(0)

        # Decide matrix for rasterization:
        # - If pdf_dpi is None => use Matrix(1,1) (native vector scale)
        # - Else compute scale = pdf_dpi / 72.0 (PyMuPDF uses 72 dpi baseline)
        if pdf_dpi is None:
            mat = fitz.Matrix(1, 1)
        else:
            scale = float(pdf_dpi) / 72.0
            mat = fitz.Matrix(scale, scale)

        pix = page.get_pixmap(matrix=mat, alpha=False)  # alpha=False to get RGB
        # Create PIL Image from pixmap bytes. pix.samples already in RGB when alpha=False.
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    else:
        # For regular raster images, open and preserve original pixel resolution
        img = Image.open(str(p))
        # Convert possible palette or CMYK to RGB for predictable downstream processing,
        # but preserve full resolution (no resizing).
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

    # ---- Apply brightness ----
    if brightness is not None and brightness != 1.0:
        enhancer_brightness = ImageEnhance.Brightness(img)
        img = enhancer_brightness.enhance(brightness)

    # ---- Apply sharpness ----
    if sharpness is not None and sharpness != 1.0:
        enhancer_sharpness = ImageEnhance.Sharpness(img)
        img = enhancer_sharpness.enhance(sharpness)

    return img


# ===========================
# Public async API
# ===========================
async def open_image(
    path_like,
    brightness: float = 1.2,
    sharpness: float = 1.5,
    pdf_dpi: Optional[int] = 300,
):
    """
    Async wrapper: reads + enhances image in a background thread.
    Use `pdf_dpi=None` to ask PyMuPDF to render with Matrix(1,1) (native scale),
    or pass an integer DPI (e.g. 300) for consistent high-resolution rasterization.
    """
    loop = asyncio.get_running_loop()
    fn = partial(_open_and_enhance_sync, path_like, brightness, sharpness, pdf_dpi)
    return await loop.run_in_executor(None, fn)
