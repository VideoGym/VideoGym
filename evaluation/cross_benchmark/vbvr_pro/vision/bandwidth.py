"""Local appearance variants through the registered video observation channel.

Labels, object counts and geometry belong to the original dictionary. Encoding
provides another observation of those same labelled pixels; its segmentation
never supplies a new answer or a replacement inventory.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from evaluation.cross_benchmark.vbvr_pro.media import video as media
from evaluation.cross_benchmark.vbvr_pro.vision.masks import _background, _paint, ink, no_added_components


_current = ContextVar("scene_observation_variants", default=None)
_raw_sources = ContextVar("scene_raw_observation_sources", default=None)
_observation_frame = ContextVar("scene_observation_frame", default=None)


class Variants:
    def __init__(self, first, final, reference):
        self.first, self.final, self.reference = first, final, reference
        originals = _raw_sources.get()
        self.semantic_first, self.semantic_final, self.semantic_reference, self.observation_fps = originals or (
            first,
            final,
            reference,
            media.DEFAULT_FPS,
        )
        self.encoder_reference = self.semantic_reference
        self.bases = [(first, "first"), (final, "final")]
        self.bases += [(frame, index) for index, frame in enumerate(reference)]
        self.encoded = None
        self.foregrounds = []
        self.equivalent = {}

    def register(self, original, resized):
        for base, role in self.bases:
            if base is original:
                self.bases.append((resized, role))
                return

    def add_equivalent(self, original, variant):
        a, b = self.resolve(original), self.resolve(variant)
        if a is not None and b is not None:

            def frame_index(role):
                return 0 if role == "first" else len(self.reference) - 1 if role == "final" else role

            # Equivalence of a fixed given cell does not make its surrounding
            # answer cells equivalent. Full-pose registrations still apply to
            # the entire image and its crops, as in the rigid-motion reader.
            base, role, region = a
            full = region == (0, 0, base.shape[1], base.shape[0])
            key = frame_index(role) if full else (frame_index(role), region)
            indices = self.equivalent.setdefault(key, [])
            index = frame_index(b[1])
            if index not in indices:
                indices.append(index)

    def _encode(self):
        if self.encoded is not None:
            return
        source = self.encoder_reference
        if not source:
            self.encoded = []
            return
        width = min(512, self.first.shape[1])
        size = (width, round(self.first.shape[0] * width / self.first.shape[1]))
        low = [cv2.resize(frame, size, interpolation=cv2.INTER_AREA) for frame in source]
        with TemporaryDirectory(prefix="vbvr-observation-") as temporary:
            path = Path(temporary) / "dictionary.mp4"
            media.encode_h264(low, path, fps=self.observation_fps)
            frames = media.decode(path).frames
        canonical = (self.first.shape[1], self.first.shape[0])
        self.encoded = [cv2.resize(frame, canonical, interpolation=cv2.INTER_LINEAR) for frame in frames]

    def resolve(self, image):
        for base, role in self.bases:
            if image.strides != base.strides or not np.shares_memory(image, base):
                continue
            offset = image.ctypes.data - base.ctypes.data
            y, remainder = divmod(offset, base.strides[0])
            x = remainder // base.strides[1]
            h, w = image.shape[:2]
            if 0 <= x <= base.shape[1] - w and 0 <= y <= base.shape[0] - h:
                return base, role, (x, y, w, h)
        return None

    def original(self, image):
        resolved = self.resolve(image)
        if resolved is None:
            return image
        base, role, (x, y, w, h) = resolved
        if role == "first" or role == 0:
            source = self.semantic_first
        elif role == "final" or role == len(self.reference) - 1:
            source = self.semantic_final
        else:
            source = self.semantic_reference[role]
        if source.shape[:2] != base.shape[:2]:
            source = cv2.resize(source, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_AREA)
        return source[y : y + h, x : x + w]

    def templates(self, image, *, region=None):
        resolved = self.resolve(image)
        if resolved is None or not self.reference:
            return
        base, role, (x, y, w, h) = resolved
        index = 0 if role == "first" else len(self.reference) - 1 if role == "final" else role
        indices = list(dict.fromkeys([index] + self.equivalent.get(index, [])))
        scoped = list(self.equivalent.get((index, (x, y, w, h)), []))
        if region is not None:
            rx, ry, rw, rh = region
            for key, variants in self.equivalent.items():
                if not isinstance(key, tuple) or key[0] != index:
                    continue
                sx, sy, sw, sh = key[1]
                if sx <= x + rx and sy <= y + ry and x + rx + rw <= sx + sw and y + ry + rh <= sy + sh:
                    scoped.extend(variants)
        scoped = [value for value in dict.fromkeys(scoped) if value not in indices]
        frame = _observation_frame.get()
        if frame is not None:
            preferred = min(frame, len(self.reference) - 1)
            scoped.sort(key=lambda value: value != preferred)
        for observed in indices:
            raw = self.reference[observed]
            yield cv2.resize(raw, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_AREA)[y : y + h, x : x + w]
        self._encode()
        if self.encoded:
            for observed in indices:
                encoded = self.encoded[observed]
                yield cv2.resize(encoded, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_AREA)[
                    y : y + h, x : x + w
                ]
        # Try each additional fixed-region frame together with its encoded
        # observation. Scanning a whole clip of raw variants before trying the
        # first matching encoded frame needlessly multiplies every given check.
        for observed in scoped:
            raw = self.reference[observed]
            yield cv2.resize(raw, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_AREA)[y : y + h, x : x + w]
            if self.encoded:
                encoded = self.encoded[observed]
                yield cv2.resize(encoded, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_AREA)[
                    y : y + h, x : x + w
                ]

    def support_box(self, image, obj, translation, actual_object):
        cached = next((entry for entry in self.foregrounds if entry[0] is image), None)
        if cached is None:
            foreground = cv2.morphologyEx(ink(image), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            _, labels, boxes, _ = cv2.connectedComponentsWithStats(foreground, 8)
            cached = (image, labels, boxes)
            self.foregrounds.append(cached)
            if len(self.foregrounds) > 4:
                self.foregrounds.pop(0)
        _, labels, boxes = cached
        selected = set()
        seeds = [(obj, translation)] + ([(actual_object, (0, 0))] if actual_object is not None else [])
        for seed, shift in seeds:
            x, y, w, h = seed.bbox
            x, y = round(x + shift[0]), round(y + shift[1])
            x0, y0, x1, y1 = max(0, x), max(0, y), min(image.shape[1], x + w), min(image.shape[0], y + h)
            if x1 <= x0 or y1 <= y0:
                continue
            local = seed.mask[y0 - y : y1 - y, x0 - x : x1 - x]
            local = cv2.dilate(local, np.ones((3, 3), np.uint8)).astype(bool)
            selected.update(np.unique(labels[y0:y1, x0:x1][local]).tolist())
        selected.discard(0)
        if not selected:
            return None
        boxes = boxes[list(selected)]
        return (
            int(boxes[:, 0].min()),
            int(boxes[:, 1].min()),
            int((boxes[:, 0] + boxes[:, 2]).max()),
            int((boxes[:, 1] + boxes[:, 3]).max()),
        )

    def pixel_size(self, image):
        resolved = self.resolve(image)
        return max(1, resolved[0].shape[1] / 512) if resolved is not None else 1


@contextmanager
def raw_observations(first, final, reference, *, observation_fps=media.DEFAULT_FPS):
    """Retain role dictionaries, raw encoder frames and the observation clock."""
    token = _raw_sources.set((first, final, reference, observation_fps))
    try:
        yield
    finally:
        _raw_sources.reset(token)


@contextmanager
def bind(first, final, reference):
    token = _current.set(Variants(first, final, reference))
    try:
        yield
    finally:
        _current.reset(token)


def original_image(image):
    context = _current.get()
    return context.original(image) if context is not None else image


def register_equivalent(original, variant):
    context = _current.get()
    if context is not None:
        context.add_equivalent(original, variant)


@contextmanager
def observation_frame(index):
    """Try the matching encoder frame first without excluding other variants."""
    token = _observation_frame.set(index)
    try:
        yield
    finally:
        _observation_frame.reset(token)


def register_resize(original, resized):
    context = _current.get()
    if context is not None:
        context.register(original, resized)


def _colour_label(actual, expected, paint, color, palette, background):
    if not paint.any():
        return False
    # Each observed paint pixel keeps its own label. A channel-wise median can
    # synthesize a colour that occurs nowhere, e.g. three wrong near-colour
    # bands whose separate channel medians happen to equal the desired paint.
    actual_rgb = actual[paint].astype(float)
    expected_rgb = expected[paint].astype(float)
    nominal = _paint(color)
    expected_ycc = cv2.cvtColor(expected, cv2.COLOR_BGR2YCrCb)[paint].astype(float)
    nominal_ycc = cv2.cvtColor(np.uint8([[nominal]]), cv2.COLOR_BGR2YCrCb)[0, 0].astype(float)
    bg_ycc = cv2.cvtColor(np.uint8([[background]]), cv2.COLOR_BGR2YCrCb)[0, 0].astype(float)
    direction = nominal_ycc[1:] - bg_ycc[1:]
    if float(direction @ direction) >= 64:
        alpha = np.sum((expected_ycc[:, 1:] - bg_ycc[1:]) * direction, axis=1) / float(direction @ direction)
    else:
        alpha = (expected_ycc[:, 0] - bg_ycc[0]) / (nominal_ycc[0] - bg_ycc[0] or 1)
    alpha = np.clip(alpha, 0.08, 1.3)
    own = np.sum((actual_rgb - expected_rgb) ** 2, axis=1)
    wrong = np.zeros(len(own), bool)
    if max(color) - min(color) > 25:
        expected_c = cv2.cvtColor(expected, cv2.COLOR_BGR2YCrCb).astype(float)[:, :, 1:] - bg_ycc[1:]
        actual_c = cv2.cvtColor(actual, cv2.COLOR_BGR2YCrCb).astype(float)[:, :, 1:] - bg_ycc[1:]
        expected_strength = cv2.dilate(np.linalg.norm(expected_c, axis=2).astype(np.float32), np.ones((3, 3), np.uint8))
        actual_strength = cv2.dilate(np.linalg.norm(actual_c, axis=2).astype(np.float32), np.ones((3, 3), np.uint8))
        wrong |= (expected_strength[paint] >= 4) & (actual_strength[paint] < 2.5)
        ec, ac = expected_c[paint], actual_c[paint]
        power = np.sum(ec**2, axis=1)
        gain = np.sum(ec * ac, axis=1) / np.maximum(1, power)
        residual = np.sum((ac - gain[:, None] * ec) ** 2, axis=1)
        actual_power = np.sum(ac**2, axis=1)
        wrong |= (power >= 16) & (actual_power >= 16) & ((gain < 0) | (residual > np.maximum(4, actual_power * 0.15)))
    # Opaque cores identify nearby discrete paints. Thin antialiased strokes
    # may lose chroma coverage without changing hue; their pointwise hue and
    # non-neutral colour evidence above remain obligatory.
    reliable = alpha >= 0.65
    for other in palette:
        if tuple(other) == tuple(color):
            continue
        counterfactual = expected_rgb + alpha[:, None] * (_paint(other) - nominal)
        wrong |= reliable & (np.sum((actual_rgb - counterfactual) ** 2, axis=1) + 1 < own)
    if float(wrong.mean()) > 0.03:
        return False
    wrong_pixels = np.zeros(paint.shape, np.uint8)
    wrong_pixels[paint] = wrong
    return no_added_components(wrong_pixels, np.zeros_like(wrong_pixels), radius=0)[0]


def _pixels_match(actual, expected, paint, color, palette, pixel_size):
    size = (max(1, round(actual.shape[1] / pixel_size)), max(1, round(actual.shape[0] / pixel_size)))
    actual = cv2.resize(actual, size, interpolation=cv2.INTER_AREA)
    expected = cv2.resize(expected, size, interpolation=cv2.INTER_AREA)
    paint = cv2.resize(paint.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0
    background = _background(expected)
    foreground = (ink(actual) | ink(expected)).astype(bool) | paint
    if not foreground.any():
        return False
    if color is not None:
        if not paint.any() or not np.any(np.linalg.norm(actual[paint].astype(float) - background, axis=1) > 8):
            return False
        if not _colour_label(actual, expected, paint, color, palette, background):
            return False
    for source, target in ((actual, expected), (expected, actual)):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        padded = np.pad(target, ((1, 1), (1, 1), (0, 0)), constant_values=255)
        difference = np.full(source.shape[:2], np.inf, np.float32)
        for dy in range(3):
            for dx in range(3):
                shifted = padded[dy : dy + source.shape[0], dx : dx + source.shape[1]]
                difference = np.minimum(difference, np.sum((source - shifted) ** 2, axis=2))
        tolerance = np.clip(np.linalg.norm(target - background, axis=2) * 0.4, 12, 65)
        wrong = (difference > tolerance**2).astype(np.uint8)
        if float(wrong[foreground].mean()) > 0.03:
            return False
        if not no_added_components(wrong, np.zeros_like(wrong), radius=0)[0]:
            return False
    return True


def complete_support_geometry(actual, expected, actual_object, expected_object, *, registered_expected=None, scale=1.0):
    """Retain attached foreground extent even if chroma loses its paint.

    Lossless gold supplies geometry; its registered observation supplies the
    pixel scale, including a crop's parent canvas or a resized motion view.
    """
    context = _current.get()
    if context is None:
        return True
    a = context.support_box(actual, actual_object, (0, 0), actual_object)
    e = context.support_box(expected, expected_object, (0, 0), expected_object)
    if a is None or e is None:
        return a is None and e is None
    # The ink support includes the resampling fringe on both sides. Two
    # observed pixels can exceed 9% of a thin rod's height even while its
    # labelled paint, shape and size all match. Keep the absolute observation
    # error separate from relative object-scale error.
    observation = expected if registered_expected is None else registered_expected
    pixel = context.pixel_size(observation)
    return all(
        abs(x - y) <= max(2 * pixel, 0.09 * y)
        for x, y in zip((a[2] - a[0], a[3] - a[1]), (scale * (e[2] - e[0]), scale * (e[3] - e[1])))
    )


def local_proof(actual, expected, obj, palette, *, position=True, actual_object=None):
    """Prove one original labelled glyph against same-size codec observations."""
    context = _current.get()
    if context is None or context.resolve(expected) is None:
        return None
    pixel = context.pixel_size(expected)
    margin = max(3, round(3 * pixel))
    x, y, w, h = obj.bbox
    offset = np.zeros(2) if position else np.subtract(actual_object.center, obj.center)
    left, top, right, bottom = x, y, x + w, y + h
    if actual_object is not None:
        ax, ay, aw, ah = actual_object.bbox
        # A local appearance proof must cover the complete observed object.
        # This retains long attached additions even when its original glyph
        # centre still matches. Codec colour fragments may enlarge the bbox;
        # their pixels are compared rather than rejected by a hard size bound.
        left, top = min(left, int(np.floor(ax - offset[0]))), min(top, int(np.floor(ay - offset[1])))
        right = max(right, int(np.ceil(ax + aw - offset[0])))
        bottom = max(bottom, int(np.ceil(ay + ah - offset[1])))
    x0, y0, x1, y1 = (
        max(0, left - margin),
        max(0, top - margin),
        min(expected.shape[1], right + margin),
        min(expected.shape[0], bottom + margin),
    )
    paint = np.zeros((y1 - y0, x1 - x0), np.uint8)
    paint[y - y0 : y - y0 + h, x - x0 : x - x0 + w] = obj.mask
    shift = round((6 if position else 2) * pixel)
    ax0, ay0 = max(0, round(x0 + offset[0]) - shift), max(0, round(y0 + offset[1]) - shift)
    ax1, ay1 = min(actual.shape[1], round(x1 + offset[0]) + shift), min(actual.shape[0], round(y1 + offset[1]) + shift)
    search = actual[ay0:ay1, ax0:ax1]
    # Local aliases describe only fixed given regions. Keep the candidate's
    # complete attached support in the comparison, including neutral tails
    # that its paint segmentation did not retain. A scoped appearance cannot
    # be borrowed when that full comparison reaches a different answer cell.
    support = context.support_box(actual, obj, offset, actual_object)
    qx0, qy0, qx1, qy1 = x0, y0, x1, y1
    if support is not None:
        sx0, sy0, sx1, sy1 = support
        qx0 = max(0, min(x0, sx0 - offset[0] - margin - shift))
        qy0 = max(0, min(y0, sy0 - offset[1] - margin - shift))
        qx1 = min(expected.shape[1], max(x1, sx1 - offset[0] + margin + shift))
        qy1 = min(expected.shape[0], max(y1, sy1 - offset[1] + margin + shift))
    for variant in context.templates(expected, region=(qx0, qy0, qx1 - qx0, qy1 - qy0)):
        template = variant[y0:y1, x0:x1]
        if min(template.shape[:2]) < 1 or search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
            continue
        response = cv2.matchTemplate(search, template, cv2.TM_SQDIFF_NORMED)
        _, _, location, _ = cv2.minMaxLoc(response)
        px, py = location
        if position and np.hypot(ax0 + px - x0, ay0 + py - y0) > 6 * pixel:
            continue
        patch = search[py : py + template.shape[0], px : px + template.shape[1]]
        if not _pixels_match(patch, template, paint, obj.color, palette, pixel):
            continue
        translation = (ax0 + px - x0, ay0 + py - y0)
        support = context.support_box(actual, obj, translation, actual_object)
        if support is not None:
            # Colour subsampling can erase a thin tail from the colour mask.
            # Complete the actor through observed foreground before claiming
            # its whole shape/size; do not crop off an attached addition.
            sx0, sy0, sx1, sy1 = support
            dx, dy = translation
            ex0 = max(0, -dx, min(x0, sx0 - dx - margin))
            ey0 = max(0, -dy, min(y0, sy0 - dy - margin))
            ex1 = min(expected.shape[1], actual.shape[1] - dx, max(x1, sx1 - dx + margin))
            ey1 = min(expected.shape[0], actual.shape[0] - dy, max(y1, sy1 - dy + margin))
            full_template = variant[ey0:ey1, ex0:ex1]
            full_actual = actual[ey0 + dy : ey1 + dy, ex0 + dx : ex1 + dx]
            full_paint = np.zeros(full_template.shape[:2], np.uint8)
            full_paint[y - ey0 : y - ey0 + h, x - ex0 : x - ex0 + w] = obj.mask
            if not _pixels_match(full_actual, full_template, full_paint, obj.color, palette, pixel):
                continue
        return {
            "paint_bbox": (x + translation[0], y + translation[1], w, h),
            "paint_mask": obj.mask,
            "location": (float(obj.center[0] + translation[0]), float(obj.center[1] + translation[1])),
        }
    return None


def match_local(actual, expected, obj, palette, *, position=True, actual_object=None):
    return local_proof(actual, expected, obj, palette, position=position, actual_object=actual_object) is not None


def distinct_support(proofs):
    if any(proof is None for proof in proofs):
        return False
    for index, a in enumerate(proofs):
        ax, ay, aw, ah = a["paint_bbox"]
        for b in proofs[index + 1 :]:
            bx, by, bw, bh = b["paint_bbox"]
            x0, y0, x1, y1 = max(ax, bx), max(ay, by), min(ax + aw, bx + bw), min(ay + ah, by + bh)
            if x1 <= x0 or y1 <= y0:
                continue
            aa = a["paint_mask"][y0 - ay : y1 - ay, x0 - ax : x1 - ax]
            bb = b["paint_mask"][y0 - by : y1 - by, x0 - bx : x1 - bx]
            if np.any(aa & bb):
                return False
    return True


def match_absence(actual, expected, unexpected_object):
    context = _current.get()
    if context is None:
        return False
    support = context.support_box(actual, unexpected_object, (0, 0), unexpected_object)
    if support is None:
        return False
    pixel = context.pixel_size(expected)
    margin = round(3 * pixel)
    x0, y0, x1, y1 = support
    x0, y0, x1, y1 = (
        max(0, x0 - margin),
        max(0, y0 - margin),
        min(actual.shape[1], x1 + margin),
        min(actual.shape[0], y1 + margin),
    )
    observed = actual[y0:y1, x0:x1]
    for variant in context.templates(expected):
        template = variant[y0:y1, x0:x1]
        if _pixels_match(observed, template, np.zeros(template.shape[:2], np.uint8), None, [], pixel):
            return True
    return False


def no_added_scene(actual, expected):
    """Preservation veto for a fixed original scene, independent of its area."""
    context = _current.get()
    if context is None:
        return False
    pixel = context.pixel_size(expected)
    size = (max(1, round(actual.shape[1] / pixel)), max(1, round(actual.shape[0] / pixel)))
    observed = ink(cv2.resize(actual, size, interpolation=cv2.INTER_AREA))
    for variant in context.templates(expected):
        allowed = ink(cv2.resize(variant, size, interpolation=cv2.INTER_AREA))
        if no_added_components(observed, allowed, radius=2)[0]:
            return True
    return False
