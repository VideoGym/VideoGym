from __future__ import annotations

from pathlib import Path


ASSET_ROOT = Path(__file__).resolve().parent


def font_path(filename: str = "OpenSans-Medium.ttf") -> Path:
    path = ASSET_ROOT / "fonts" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing packaged font asset: {path}")
    return path


def render_font(size: int, *, bold: bool = False):
    """Bundled DejaVu Sans renderer font, independent of host font discovery."""
    from PIL import ImageFont
    return ImageFont.truetype(str(font_path("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


__all__ = ["ASSET_ROOT", "font_path", "render_font"]
