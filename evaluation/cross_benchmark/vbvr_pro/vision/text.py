"""Optional local OCR for literal answer text; never a correctness judge."""

import re
import shutil
import os
import subprocess

import cv2


def read_count(image):
    executable = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if executable is None:
        return None
    ok, png = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Cannot encode the answer text observation")
    process = subprocess.run(
        [executable, "stdin", "stdout", "--psm", "7", "-l", "eng"],
        input=png.tobytes(),
        capture_output=True,
        timeout=10,
        check=False,
    )
    if process.returncode:
        raise RuntimeError("Local OCR failed to execute")
    text = process.stdout.decode("utf-8", errors="replace").strip()
    matched = re.fullmatch(r"[Cc]ount\s*:?\s*(\d+)\s*[.!]?", text)
    return int(matched.group(1)) if matched else None
