"""Task-selected Prompt Paraphrase context, inert outside visual evaluation.

Renderers and task-owned observers share the declared symbols and display
coordinates. Solvers, RNGs and logical state remain independent of the view;
observers never fit a mapping to candidate content.
"""

from contextlib import contextmanager
from contextvars import ContextVar

_ACTIVE = ContextVar("videogym_prompt_paraphrase", default=None)


def current_prompt_paraphrase():
    return _ACTIVE.get()


@contextmanager
def use_prompt_paraphrase(plan=None):
    token = _ACTIVE.set(plan)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def active_profile(name):
    return (current_prompt_paraphrase() or {}).get('profile') == name


def draw_column(draw, box, color, outline):
    """A height-bearing signal mast, with a solid endpoint and narrow stem."""
    x0, y0, x1, y1 = map(float, box)
    w, h = x1-x0, y1-y0
    cx = (x0+x1)/2
    draw.rectangle((cx-w*.11,y0+h*.12,cx+w*.11,y1),fill=color,outline=outline,width=1)
    draw.polygon([(cx,y0),(x1,y0+h*.14),(cx,y0+h*.28),(x0,y0+h*.14)],fill=color,outline=outline,width=2)
    draw.polygon([(cx-w*.3,y1-h*.1),(cx+w*.3,y1-h*.1),(x1,y1),(x0,y1)],fill=color,outline=outline,width=2)


def draw_crate(draw, box, color, outline):
    """A broad double-wing platform whose exact span encodes disk size."""
    x0,y0,x1,y1=map(float,box);w=x1-x0;h=y1-y0;cx=(x0+x1)/2
    draw.polygon([(x0,y0+h*.5),(x0+w*.18,y0),(cx-w*.12,y0+h*.25),(cx,y0),(cx+w*.12,y0+h*.25),(x1-w*.18,y0),(x1,y0+h*.5),(x1-w*.18,y1),(cx+w*.12,y1-h*.25),(cx,y1),(cx-w*.12,y1-h*.25),(x0+w*.18,y1)],fill=color,outline=outline,width=2)
    draw.line((x0+w*.15,y0+h*.5,x1-w*.15,y0+h*.5),fill=outline,width=2)


def draw_gem(draw, box, color, outline=(44,48,54)):
    """A four-lobed signal token, preserving its bounding height and colour."""
    x0,y0,x1,y1=map(float,box);w=x1-x0;h=y1-y0;cx=(x0+x1)/2;cy=(y0+y1)/2
    for rect in ((cx-w*.18,y0,cx+w*.18,cy),(cx-w*.18,cy,cx+w*.18,y1),(x0,cy-h*.18,cx,cy+h*.18),(cx,cy-h*.18,x1,cy+h*.18)):
        draw.ellipse(rect,fill=color,outline=outline,width=2)
    draw.ellipse((cx-w*.2,cy-h*.2,cx+w*.2,cy+h*.2),fill=color,outline=outline,width=2)


def draw_crown(draw, box, color):
    """Queen role as an eight-direction compass, with no directional preference."""
    import math
    x0,y0,x1,y1=map(float,box);cx=(x0+x1)/2;cy=(y0+y1)/2;r=min(x1-x0,y1-y0)/2
    points=[]
    for i in range(16):
        angle=i*math.pi/8;radius=r if i%2==0 else r*.38
        points.append((cx+radius*math.cos(angle),cy+radius*math.sin(angle)))
    draw.polygon(points,fill=color)
    draw.ellipse((cx-r*.22,cy-r*.22,cx+r*.22,cy+r*.22),fill=(249,237,202))


def draw_bevel(draw,box,color,edge):
    """Occupancy token as a spool with four unmistakable cell lobes."""
    x0,y0,x1,y1=map(float,box);w=x1-x0;h=y1-y0
    draw.polygon([(x0,y0),(x0+w*.3,y0),(x0+w*.5,y0+h*.25),(x1-w*.3,y0),(x1,y0),(x1,y1),(x1-w*.3,y1),(x0+w*.5,y1-h*.25),(x0+w*.3,y1),(x0,y1)],fill=color,outline=edge,width=2)
    draw.line((x0+w*.5,y0+h*.3,x0+w*.5,y1-h*.3),fill=edge,width=2)
