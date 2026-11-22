"""Validate and resume complete candidate videos."""

from pathlib import Path


def valid_video(path, frames):
    """Decode the entire candidate before accepting a resume or completion."""
    import cv2

    path = Path(path)
    if not path.is_file():
        return False
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened() or abs(capture.get(cv2.CAP_PROP_FPS) - 16) > 0.05:
            return False
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != (512, 512, 3):
                return False
            count += 1
        return count == frames
    finally:
        capture.release()


def pending_inputs(output, rows, timings):
    output = Path(output)
    return [
        row
        for row in rows
        if not valid_video(output / row["id"] / "prediction.webm", timings[row["id"]]["frames"])
    ]
