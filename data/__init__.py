"""Data generation and export pipeline."""

from __future__ import annotations

import hashlib
import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path


_MINIMUM_PYTHON = (3, 12)
_VISUAL_RUNTIME = (
    ("numpy", "numpy", (2, 0, 2), (3, 0, 0)),
    ("cv2", "opencv-python", (4, 13, 0, 92), (6, 0, 0)),
    ("PIL", "Pillow", (10, 4, 0), (13, 0, 0)),
)
_PILLOW_FEATURES = ("freetype2", "raqm", "harfbuzz", "fribidi")
_FONT_PROBE_SHA256 = (
    "bc85c6ffc4f78abe5cec4ead823ac0d8678a760787a10677cf537f68a980deff"
)


def _release_tuple(raw: str, distribution: str) -> tuple[int, ...]:
    parts = raw.split(".")
    if not parts or any(not part.isdecimal() for part in parts):
        raise RuntimeError(
            f"VideoGym requires a final numeric {distribution} release; found {raw}."
        )
    return tuple(int(part) for part in parts)


def _compare_release(
    actual: tuple[int, ...],
    boundary: tuple[int, ...],
) -> int:
    width = max(len(actual), len(boundary))
    padded_actual = actual + (0,) * (width - len(actual))
    padded_boundary = boundary + (0,) * (width - len(boundary))
    return (padded_actual > padded_boundary) - (padded_actual < padded_boundary)


def _validate_visual_runtime() -> None:
    if sys.version_info < _MINIMUM_PYTHON:
        actual = ".".join(str(value) for value in sys.version_info[:3])
        raise RuntimeError(
            "VideoGym visual generation and verification require Python 3.12 or "
            f"newer; found {actual}. Older runtimes can change raster replay "
            "and silently reject valid reference media."
        )
    for module_name, distribution, minimum, maximum in _VISUAL_RUNTIME:
        import_module(module_name)
        try:
            actual_raw = distribution_version(distribution)
        except PackageNotFoundError as error:
            if distribution != "opencv-python":
                raise RuntimeError(f"VideoGym requires the {distribution} distribution.") from error
            # Headless environments use the same cv2 API without desktop GUI libraries.
            distribution = "opencv-python-headless"
            try:
                actual_raw = distribution_version(distribution)
            except PackageNotFoundError as headless_error:
                raise RuntimeError("VideoGym requires opencv-python or opencv-python-headless.") from headless_error
        actual = _release_tuple(actual_raw, distribution)
        if _compare_release(actual, minimum) < 0 or _compare_release(actual, maximum) >= 0:
            minimum_text = ".".join(str(value) for value in minimum)
            maximum_text = ".".join(str(value) for value in maximum)
            raise RuntimeError(
                f"VideoGym requires {distribution}>={minimum_text},<{maximum_text}; "
                f"found {actual_raw}."
            )
    pillow_features = import_module("PIL.features")
    missing_features = [
        feature
        for feature in _PILLOW_FEATURES
        if not pillow_features.check(feature)
    ]
    if missing_features:
        raise RuntimeError(
            "VideoGym requires a Pillow build with "
            f"{', '.join(_PILLOW_FEATURES)}; missing {', '.join(missing_features)}. "
            "Pillow native text-layout features are part of the visual contract."
        )
    image_font = import_module("PIL.ImageFont")
    font_path = (
        Path(__file__).resolve().parent
        / "common"
        / "assets"
        / "fonts"
        / "OpenSans-Medium.ttf"
    )
    probe_font = image_font.truetype(font_path.as_posix(), size=20)
    probe_digest = hashlib.sha256(bytes(probe_font.getmask("12345"))).hexdigest()
    if (
        probe_font.layout_engine != image_font.Layout.RAQM
        or probe_digest != _FONT_PROBE_SHA256
    ):
        raise RuntimeError(
            "VideoGym Pillow/font raster fingerprint does not match the supported "
            "RAQM visual runtime. Regenerate the environment; do not relax "
            "verifier thresholds."
        )


_validate_visual_runtime()
