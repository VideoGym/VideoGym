"""Runtime capture for text actually sent to the raster renderer.

The capture is intentionally local to one generated record.  It observes PIL
``draw.text`` calls rather than inferring glyphs from task names or semantic
metadata, so manifests cannot claim labels that a renderer never attempted to
show.  Family visual QA remains responsible for occlusion and readability.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, MutableSet, Optional, Tuple

from PIL import ImageDraw


GlyphCall = Tuple[str, Optional[Tuple[float, float, float, float]], int]


@contextmanager
def capture_pil_text() -> Iterator[List[GlyphCall]]:
    """Capture every non-empty PIL text run and its best-effort bounding box."""

    calls: List[GlyphCall] = []
    # Tag PIL images with a capture-local token instead of using ``id(image)``.
    # Renderers often release a temporary PIL frame immediately after turning
    # it into an ndarray, so CPython can reuse the id for a later frame and
    # manufacture a false text collision.  Keeping every video frame alive
    # would avoid that bug but needlessly multiplies four-worker peak memory.
    capture_marker = object()
    token_attribute = "_videogym_rendered_glyph_capture_token"
    next_image_token = 0
    fallback_refs: Dict[int, Any] = {}
    fallback_tokens: Dict[int, int] = {}
    original = ImageDraw.ImageDraw.text

    def tracked(draw: ImageDraw.ImageDraw, xy: Any, text: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal next_image_token
        value = str(text)
        if value.strip():
            bbox = None
            try:
                raw = draw.textbbox(xy, value, font=kwargs.get("font"), anchor=kwargs.get("anchor"))
                bbox = tuple(float(item) for item in raw)
            except Exception:
                bbox = None
            image = getattr(draw, "_image", getattr(draw, "im", draw))
            tagged = getattr(image, token_attribute, None)
            if (
                isinstance(tagged, tuple)
                and len(tagged) == 2
                and tagged[0] is capture_marker
            ):
                image_token = int(tagged[1])
            else:
                image_token = next_image_token
                next_image_token += 1
                try:
                    setattr(image, token_attribute, (capture_marker, image_token))
                except Exception:
                    # Non-PIL draw backends may reject attributes.  Strongly
                    # retain only those rare objects so their ids cannot be
                    # reused within this capture scope.
                    image_key = id(image)
                    fallback_refs.setdefault(image_key, image)
                    image_token = fallback_tokens.setdefault(image_key, image_token)
            calls.append((value, bbox, image_token))
        return original(draw, xy, text, *args, **kwargs)

    ImageDraw.ImageDraw.text = tracked
    try:
        yield calls
    finally:
        ImageDraw.ImageDraw.text = original


def rendered_glyph_contract(calls: List[GlyphCall]) -> tuple[list[str], Dict[str, object]]:
    """Build an exact text-run inventory plus a bounded character schema."""

    text_runs = sorted({" ".join(value.split()) for value, _bbox, _image in calls if value.strip()})
    characters: MutableSet[str] = set()
    for run in text_runs:
        characters.update(character for character in run if not character.isspace())
    inventory = sorted(set(text_runs) | characters)
    if not inventory:
        raise ValueError("renderer produced no captured text glyphs")
    invalid_runs = [
        run
        for run in text_runs
        if "\n" in run or len(run) > 32 or len(run.split()) > 4
    ]
    if invalid_runs:
        raise ValueError(f"rendered text exceeds the bounded low-text schema: {invalid_runs}")
    return inventory, {
        "kind": "runtime_pil_text_capture_v1",
        "capture_source": "actual_draw_text_calls",
        "text_runs": text_runs,
        "glyph_characters": sorted(characters),
        "max_text_run_characters": 32,
        "max_words_per_text_run": 4,
        "free_form_text_allowed": False,
    }


def text_bbox_collisions(calls: List[GlyphCall]) -> list[Dict[str, object]]:
    """Return non-trivial overlaps between separately drawn text runs."""

    collisions: list[Dict[str, object]] = []
    for index, (left_text, left_box, left_image) in enumerate(calls):
        if left_box is None:
            continue
        for right_text, right_box, right_image in calls[index + 1 :]:
            if right_box is None or left_image != right_image:
                continue
            x0 = max(left_box[0], right_box[0]); y0 = max(left_box[1], right_box[1])
            x1 = min(left_box[2], right_box[2]); y1 = min(left_box[3], right_box[3])
            if x1 - x0 <= 1 or y1 - y0 <= 1:
                continue
            # Repeated redraws of the same run at the same place are expected
            # only across separate images, which have distinct image ids.
            collisions.append({
                "first": left_text,
                "second": right_text,
                "intersection": [x0, y0, x1, y1],
            })
    return collisions


__all__ = ["GlyphCall", "capture_pil_text", "rendered_glyph_contract", "text_bbox_collisions"]
