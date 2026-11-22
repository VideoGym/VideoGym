"""CPU OCR for small printed digits whose raster contours are ambiguous."""

from __future__ import annotations

import io
import os
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageOps


def read_printed_digit(patch: np.ndarray) -> dict:
    """Read the complete decimal vocabulary without an expected-answer prior.

    Two single-character segmentation modes must agree when both read a digit.
    An unrecognized or conflicting observation remains explicit in the result.
    """
    engine = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if not engine:
        raise RuntimeError("Graph degree verification requires Tesseract 5 with English tessdata")
    image = Image.fromarray(np.asarray(patch, dtype=np.uint8)).convert("L")
    image = ImageOps.expand(image.resize((170, 170)), border=30, fill=255)
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    readings = []
    for mode in (10, 13):
        result = subprocess.run(
            [engine, "stdin", "stdout", "--psm", str(mode),
             "-c", "tessedit_char_whitelist=0123456789"],
            input=encoded.getvalue(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True,
        )
        text = result.stdout.decode().strip()
        readings.append(int(text) if len(text) == 1 and text.isascii() and text.isdigit() else None)
    values = {value for value in readings if value is not None}
    return {"digit": next(iter(values)) if len(values) == 1 else None,
            "readings": readings, "engine": "tesseract"}
