"""Opt-in, role-aware scene skins shared by paired visual evaluation tasks.

The context changes rendering only. Source graphs, grids, object identities,
colors with rule meanings, simulation and solutions are never modified.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math


_ACTIVE = ContextVar("videogym_visual_style", default=None)


def current_style():
    return _ACTIVE.get()


@contextmanager
def use_style(name):
    token = _ACTIVE.set(STYLES[name] if name is not None else None)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def _mix(color, other, amount):
    return tuple(round((1 - amount) * a + amount * b) for a, b in zip(color, other))


@dataclass(frozen=True)
class VisualStyle:
    name: str
    font: str
    node_shape: str
    background: tuple[int, int, int]
    background_name: str
    edge_name: str
    description: str

    def surface(self, draw, bounds, color):
        x0, y0, x1, y1 = bounds
        draw.rectangle(bounds, fill=color)
        if self.name == "beacons":
            draw.rounded_rectangle((x0+14,y0+14,x1-14,y1-14),radius=18,
                                   outline=(211,196,222),width=2)
            return
        if self.name == "switchboard":
            for x in range(int(x0) + 16, int(x1), 32):
                for y in range(int(y0) + 16, int(y1), 32):
                    draw.ellipse((x-1, y-1, x+1, y+1), fill=(209, 218, 201))
            draw.rectangle((x0+12,y0+12,x1-12,y1-12), outline=(178,191,166), width=2)
            return
        light = _mix(color, (255, 255, 255), .10)
        dark = _mix(color, (70, 75, 80), .035)
        if self.name == "cut_paper":
            # Quiet, deterministic paper fibers; no random latent state.
            for y in range(int(y0) + 3, int(y1), 7):
                for x in range(int(x0) + (y % 3), int(x1) - 2, 13):
                    draw.line((x, y, min(x + 3, x1), y), fill=dark, width=.5)
        else:
            for y in range(int(y0) + 4, int(y1), 11):
                draw.line((x0, y, x1, y), fill=light, width=.5)
                draw.line((x0, y + 1, x1, y + 1), fill=dark, width=.5)

    def token(self, draw, box, fill, outline, *, width=2):
        x0, y0, x1, y1 = box
        if self.name == "beacons":
            cx,cy=(x0+x1)/2,(y0+y1)/2
            radius=(x1-x0)/2
            pts=[(cx+radius*(1 if i%2==0 else .8)*math.cos(i*math.pi/6),
                  cy+radius*(1 if i%2==0 else .8)*math.sin(i*math.pi/6)) for i in range(12)]
            draw.polygon(pts,fill=(232,219,242),outline=(129,108,153),width=2)
            draw.ellipse((x0+4,y0+4,x1-4,y1-4),fill=fill,outline=outline,width=1)
        elif self.name == "switchboard":
            size = x1-x0
            for f in (.28, .72):
                x, y = x0+f*size, y0+f*size
                draw.line((x,y0-3,x,y1+3),fill=(177,189,187),width=3)
                draw.line((x0-3,y,x1+3,y),fill=(177,189,187),width=3)
            draw.rectangle(box, fill=(223,229,218), outline=(22,40,47), width=2)
            draw.rectangle((x0+3,y0+3,x1-3,y1-3),fill=fill,outline=(124,148,148),width=1)
        elif self.name == "cut_paper":
            # Raised folded-paper square; identity lives in its label or color.
            draw.rounded_rectangle(box, radius=(x1 - x0) * .16,
                                   fill=fill, outline=outline, width=width)
            inset = max(2, (x1 - x0) * .11)
            draw.line((x0 + inset, y1 - inset, x1 - inset, y1 - inset,
                       x1 - inset, y0 + inset), fill=_mix(fill, outline, .35), width=1.5)
            draw.line((x0 + inset, y1 - inset, x0 + inset, y0 + inset,
                       x1 - inset, y0 + inset), fill=_mix(fill, (255, 255, 255), .75), width=1.5)
        else:
            # Eight-sided machined cap, retaining the original center and extent.
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
            points = [(cx + rx * math.cos(math.pi / 8 + i * math.pi / 4),
                       cy + ry * math.sin(math.pi / 8 + i * math.pi / 4)) for i in range(8)]
            draw.polygon(points, fill=fill, outline=outline, width=width)
            inner = [(cx + .79 * (x - cx), cy + .79 * (y - cy)) for x, y in points]
            draw.line(inner[:5], fill=_mix(fill, (255, 255, 255), .65), width=1.5)
            draw.line(inner[4:] + inner[:1], fill=_mix(fill, outline, .40), width=1.5)

    def edge(self, draw, points, width):
        # A connector has one semantic edge regardless of its dash/armor count.
        (x0, y0), (x1, y1) = points
        length = math.hypot(x1 - x0, y1 - y0)
        if length == 0:
            return
        if self.name == "beacons":
            draw.line(points,fill=(224,211,233),width=width+4)
            draw.line(points,fill=(133,113,155),width=max(2,width-1))
            return
        if self.name == "switchboard":
            draw.line(points, fill=(218,224,205), width=width+7)
            draw.line(points, fill=(185,201,173), width=width+3)
            draw.line(points, fill=(71,115,108), width=max(1,width-1))
            return
        if self.name == "machined":
            draw.line(points, fill=(210, 242, 250), width=width + 3)
            dash_color, step, on = (30, 153, 183), 15, 9
        else:
            dash_color, step, on = (104, 46, 126), 17, 10
        for start in range(0, math.ceil(length), step):
            end = min(start + on, length)
            segment = ((x0 + (x1 - x0) * start / length, y0 + (y1 - y0) * start / length),
                       (x0 + (x1 - x0) * end / length, y0 + (y1 - y0) * end / length))
            draw.line(segment, fill=dash_color, width=width + 1)

    def maze_cell(self, draw, box, *, wall, fill, outline):
        x0, y0, x1, y1 = box
        draw.rectangle(box, fill=fill, outline=outline, width=1)
        if wall:
            light = _mix(fill, (190, 200, 210), .25)
            if self.name == "cut_paper":
                step = max(6, int((y1 - y0) / 3))
                for index, y in enumerate(range(int(y0) + step, int(y1), step)):
                    draw.line((x0 + 1, y, x1 - 1, y), fill=light, width=1)
                    x = x0 + (x1 - x0) * (.3 if index % 2 else .65)
                    draw.line((x, y - step + 1, x, y), fill=light, width=1)
            else:
                draw.line((x0 + 3, y1 - 3, x0 + 3, y0 + 3, x1 - 3, y0 + 3),
                          fill=light, width=1.5)
                for x in (x0 + 5, x1 - 5):
                    for y in (y0 + 5, y1 - 5):
                        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=light)
        else:
            # The center stays white and readable under routes and mechanisms.
            draw.line((x0 + 2, y1 - 2, x1 - 2, y1 - 2), fill=(229, 231, 234), width=1)

    def key(self, draw, box, fill, outline):
        x0, y0, x1, y1 = box
        size = x1 - x0
        if self.name == "cut_paper":
            # A chunky vertical key with a circular bow and asymmetric teeth.
            draw.ellipse((x0 + .10 * size, y0 + .06 * size, x0 + .90 * size, y0 + .74 * size),
                         fill=fill, outline=outline, width=2)
            draw.ellipse((x0 + .30 * size, y0 + .23 * size, x0 + .70 * size, y0 + .54 * size),
                         fill=(255, 255, 255), outline=outline, width=1)
            draw.rectangle((x0 + .38 * size, y0 + .61 * size, x0 + .66 * size, y0 + .94 * size),
                           fill=fill, outline=outline, width=1)
            draw.rectangle((x0 + .59 * size, y0 + .73 * size, x0 + .88 * size, y0 + .87 * size),
                           fill=fill, outline=outline, width=1)
        else:
            # A keycard with an unmistakable key cutout, same functional object.
            draw.rounded_rectangle((x0 + .15 * size, y0 + .13 * size, x1 - .15 * size, y1 - .13 * size),
                                   radius=.08 * size, fill=fill, outline=outline, width=2)
            draw.ellipse((x0 + .28 * size, y0 + .28 * size, x0 + .53 * size, y0 + .53 * size),
                         outline=outline, width=2)
            draw.line((x0 + .48 * size, y0 + .49 * size, x0 + .72 * size, y0 + .73 * size,
                       x0 + .76 * size, y0 + .63 * size), fill=outline, width=2)

    def gate(self, draw, box, fill, outline):
        x0, y0, x1, y1 = box
        size = x1 - x0
        draw.rounded_rectangle((x0 + .1 * size, y0 + .1 * size, x1 - .1 * size, y1 - .1 * size),
                               radius=.06 * size, fill=fill, outline=outline, width=2)
        if self.name == "cut_paper":
            for x in (.30, .50, .70):
                draw.line((x0 + x * size, y0 + .18 * size, x0 + x * size, y0 + .82 * size),
                          fill=outline, width=2)
        else:
            draw.line((x0 + .21 * size, y0 + .23 * size, x0 + .79 * size, y0 + .77 * size),
                      fill=outline, width=2)
            draw.line((x0 + .79 * size, y0 + .23 * size, x0 + .21 * size, y0 + .77 * size),
                      fill=outline, width=2)
        draw.ellipse((x0 + .40 * size, y0 + .34 * size, x0 + .60 * size, y0 + .55 * size),
                     fill=outline)
        draw.rectangle((x0 + .46 * size, y0 + .50 * size, x0 + .54 * size, y0 + .68 * size), fill=outline)


STYLES = {
    "beacons": VisualStyle("beacons", "OpenSans-Medium.ttf", "labelled beacon rosettes",
                            (253,249,254), "pale lilac", "violet connector",
                            "a light beacon field with rosette vertices and fine violet edges"),
    "switchboard": VisualStyle("switchboard", "DejaVuSans.ttf", "square plug terminals",
                                (251,248,234), "ivory route-map", "insulated cable",
                                "staggered terminal banks, plugged sockets and insulated graph cables"),
    "cut_paper": VisualStyle("cut_paper", "OpenSans-Medium.ttf", "rounded square tiles",
                              (174, 211, 192), "sage-green", "purple dashed connector",
                              "green paper canvas, square tiles, dashed edges, brickwork and chunky keys"),
    "machined": VisualStyle("machined", "DejaVuSans.ttf", "octagonal tiles",
                             (18, 34, 57), "navy-blue", "cyan armored cable",
                             "navy canvas, octagonal caps, cyan cables, metal walls and keycards"),
}
