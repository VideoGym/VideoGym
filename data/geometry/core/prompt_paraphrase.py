"""Task-authored geometric scene primitives, independent of problem sampling.

Physical carriers expose the same mathematical centrelines and point centres.
The ornamental bodies supply neither measurements nor solution constructions.
"""
import math
from PIL import ImageFont
from data.common.prompt_paraphrase import current_prompt_paraphrase
from data.common.render_primitives import ScaledDraw, stroke_polyline


SCENES = {
    'midpoint': ('survey_bar', (246, 251, 246)),
    'perpendicular_bisector': ('beacon_baseline', (255, 252, 240)),
    'angle_bisector': ('light_beams', (247, 246, 255)),
    'circumcenter': ('antenna_links', (246, 250, 255)),
    'incenter': ('masonry_courtyard', (252, 248, 241)),
    'orthocenter': ('tension_cables', (247, 252, 250)),
    'diagonal_intersection': ('glass_frame', (247, 250, 253)),
    'orthogonal_projection': ('guide_gauge', (255, 251, 245)),
    'segment_trisection': ('measuring_strap', (252, 247, 250)),
    'closest_pair': ('star_field', (250, 248, 255)),
}


STYLES = {
    'midpoint': ('square', 'numbers', (40,93,61), 'forest-green'),
    'perpendicular_bisector': ('triangle', 'letters', (89,63,119), 'violet'),
    'angle_bisector': ('pentagon', 'letters', (54,72,122), 'slate-blue'),
    'circumcenter': ('square', 'numbers', (38,87,62), 'forest-green'),
    'incenter': ('pentagon', 'letters', (99,104,45), 'olive'),
    'orthocenter': ('triangle', 'letters', (83,63,111), 'violet'),
    'diagonal_intersection': ('hexagon', 'numbers', (47,66,108), 'slate-blue'),
    'orthogonal_projection': ('diamond', 'numbers', (86,64,116), 'violet'),
    'segment_trisection': ('hexagon', 'letters', (39,91,65), 'forest-green'),
    'closest_pair': ('star', 'numbers', (70,63,105), 'violet'),
}


def prompt_paraphrase_style(task):
    shape,labels,colour,name=STYLES[task]
    return {'candidate_shape':shape,'label_set':labels,'given_colour':list(colour),'given_colour_name':name}


def given_colour(colour):
    if workbench_active() and colour is not None and max(colour)<110 and max(colour)-min(colour)<18:
        return STYLES[(current_prompt_paraphrase() or {})['task']][2]
    return colour


def workbench_active():
    return (current_prompt_paraphrase() or {}).get('profile') == 'geometry_workbench'


def scene():
    return SCENES[(current_prompt_paraphrase() or {})['task']][0]


def draw_workbench(draw, width, height, candidates=(), radius=10):
    kind, background=SCENES[(current_prompt_paraphrase() or {})['task']]
    draw.rectangle((0,0,width,height),fill=background)
    # These sparse edge treatments have no repeated spacing usable as a ruler.
    if kind in ('survey_bar','guide_gauge'):
        draw.rectangle((0,height-19,width,height),fill=(218,225,213))
        draw.line(((0,height-19),(width,height-19)),fill=(165,176,162),width=2)
    elif kind=='masonry_courtyard':
        draw.polygon([(0,0),(88,0),(0,66)],fill=(237,229,207))
        draw.polygon([(width,height),(width-71,height),(width,height-93)],fill=(237,229,207))
    elif kind=='glass_frame':
        draw.polygon([(0,0),(width,0),(0,18)],fill=(225,230,239))
        draw.polygon([(width,height),(0,height),(width,height-12)],fill=(225,230,239))
    elif kind=='tension_cables':
        for x,y in ((12,15),(width-19,9),(width-11,height-21),(18,height-13)):
            draw.rectangle((x-4,y-4,x+4,y+4),fill=(158,174,163))
    elif kind=='measuring_strap':
        draw.line(((0,8),(width,8)),fill=(228,213,228),width=3)
        draw.line(((0,height-9),(width,height-9)),fill=(228,213,228),width=3)

    draw_candidate_supports(draw,candidates,radius)


def draw_candidate_supports(draw,candidates,radius):
    ss=draw.factor if isinstance(draw,ScaledDraw) else 1
    raw=draw.raw if isinstance(draw,ScaledDraw) else draw
    kind=scene()
    for point in candidates:
        cx,cy=round(point.x*ss),round(point.y*ss);r=round(radius*ss)
        if kind=='star_field':
            outer=round(r*1.65)
            star=[(cx,cy-outer),(cx+3*ss,cy-3*ss),(cx+outer,cy),
                  (cx+3*ss,cy+3*ss),(cx,cy+outer),(cx-3*ss,cy+3*ss),
                  (cx-outer,cy),(cx-3*ss,cy-3*ss)]
            raw.polygon(star,fill=(170,158,196),outline=(95,81,122),width=ss)
        elif kind=='antenna_links':
            raw.line((cx-r-4*ss,cy+r+2*ss,cx+r+4*ss,cy+r+2*ss),fill=(122,132,149),width=2*ss)
        elif kind=='masonry_courtyard':
            raw.line((cx,cy+r,cx,cy+r+7*ss),fill=(105,101,84),width=2*ss)
        elif kind=='guide_gauge':
            raw.rectangle((cx-r-3*ss,cy-r//2,cx+r+3*ss,cy+r//2),fill=(198,188,156))
        elif kind=='light_beams':
            raw.rectangle((cx-r-2*ss,cy-r//2,cx+r+2*ss,cy+r//2),fill=(190,179,215))


def _segments(points):
    for a,b in zip(points,points[1:]):
        dx,dy=b[0]-a[0],b[1]-a[1]
        length=math.hypot(dx,dy)
        if length>0:yield a,b,(-dy/length,dx/length),length


def draw_survey_rail(draw, points, color, width):
    """Render task-specific given carriers below the exact source centreline."""
    if color is None or max(color)-min(color)>18 or max(color)>100:return
    kind=scene()
    if kind=='light_beams':
        # Each ray's carrier widens away from the common vertex. Only the
        # centreline is a geometric ray; neither beam boundary is a new ray.
        vertex=points[1] if len(points)==3 else points[0]
        for tip in (points[0],points[-1]):
            dx,dy=tip[0]-vertex[0],tip[1]-vertex[1]
            length=math.hypot(dx,dy)
            if not length:continue
            nx,ny=-dy/length,dx/length
            draw.polygon([vertex,(tip[0]+nx*18,tip[1]+ny*18),(tip[0]-nx*18,tip[1]-ny*18)],fill=(224,216,245))
        return
    if kind=='beacon_baseline':
        # A broad translucent connection identifies the same finite segment.
        for a,b,n,length in _segments(points):
            for t in range(0,int(length),13):
                u=min(length,t+7)
                draw.line(((a[0]+(b[0]-a[0])*t/length,a[1]+(b[1]-a[1])*t/length),
                           (a[0]+(b[0]-a[0])*u/length,a[1]+(b[1]-a[1])*u/length)),fill=given_colour(color),width=3)
        return True
    if kind=='antenna_links':
        stroke_polyline(draw,points,fill=(218,224,239),width=width+7)
        for x,y in points:
            draw.polygon([(x-10,y+13),(x+10,y+13),(x,y-16)],fill=(207,212,230),outline=(94,103,124),width=2)
            draw.ellipse((x-4,y-4,x+4,y+4),fill=(248,249,255),outline=(70,79,99),width=2)
        return
    if kind=='masonry_courtyard':
        stroke_polyline(draw,points,fill=(177,170,145),width=width+20)
        stroke_polyline(draw,points,fill=(235,222,188),width=width+15)
        for a,b,n,length in _segments(points):
            # Decorative joints are deliberately irregular and unnumbered.
            for fraction in (0.17,0.43,0.74,0.88):
                x=a[0]+(b[0]-a[0])*fraction;y=a[1]+(b[1]-a[1])*fraction
                draw.line(((x-n[0]*9,y-n[1]*9),(x+n[0]*9,y+n[1]*9)),fill=(190,181,153),width=2)
        return
    if kind=='tension_cables':
        stroke_polyline(draw,points,fill=(207,220,207),width=width+9)
        for x,y in points:
            draw.rectangle((x-7,y-7,x+7,y+7),fill=(153,171,156),outline=(68,82,73),width=2)
            draw.line(((x-3,y-3),(x+3,y+3)),fill=(240,249,239),width=2)
        return
    if kind=='glass_frame':
        if len(points)>3 and points[0]==points[-1]:
            draw.polygon(points,fill=(229,235,245))
        stroke_polyline(draw,points,fill=(130,147,160),width=width+24)
        stroke_polyline(draw,points,fill=(230,235,239),width=width+17)
        stroke_polyline(draw,points,fill=(179,193,205),width=width+10)
        return
    if kind=='measuring_strap':
        stroke_polyline(draw,points,fill=(151,137,153),width=width+15)
        stroke_polyline(draw,points,fill=(230,211,230),width=width+10)
        # No ticks, divisions or repeated marks: the 1/3 point is not given.
        return
    if kind=='guide_gauge':
        stroke_polyline(draw,points,fill=(199,193,169),width=width+20)
        stroke_polyline(draw,points,fill=(248,239,211),width=width+13)
        for x,y in (points[0],points[-1]):
            draw.rectangle((x-7,y-7,x+7,y+7),fill=(177,169,146),outline=(72,75,68),width=2)
        return
    if kind=='survey_bar':
        stroke_polyline(draw,points,fill=(166,179,158),width=width+22)
        stroke_polyline(draw,points,fill=(230,239,217),width=width+16)


def draw_survey_markers(cls, draw, candidates, radius, highlight_label=None, progress=1.0):
    ss=draw.factor if isinstance(draw,ScaledDraw) else 1
    raw=draw.raw if isinstance(draw,ScaledDraw) else draw
    style=prompt_paraphrase_style((current_prompt_paraphrase() or {})['task'])
    shape=style['candidate_shape']
    font_size=9 if shape=='triangle' else 10 if shape in ('square','diamond') else cls.CANDIDATE_LABEL_FONT_SIZE
    font=ImageFont.load_default(round(font_size*ss))
    active=set(str(highlight_label or '').upper())
    def blend(a,b,t):return tuple(round(x+(y-x)*t) for x,y in zip(a,b))
    for point in sorted(candidates,key=lambda p:p.label):
        cx,cy=round(point.x*ss),round(point.y*ss);r=round(radius*ss)
        fraction=progress if point.label.upper() in active else 0.0
        fill=blend((246,247,238),cls.CANDIDATE_HIGHLIGHT_FILL,fraction)
        edge=blend((48,56,51),cls.CANDIDATE_HIGHLIGHT_COLOR,fraction)
        sides={'triangle':3,'square':4,'diamond':4,'pentagon':5,'hexagon':6,'star':6}[shape]
        phase=math.pi/4 if shape=='square' else 0 if shape=='diamond' else -math.pi/2
        polygon=[(round(cx+r*math.cos(phase+i*2*math.pi/sides)),
                  round(cy+r*math.sin(phase+i*2*math.pi/sides))) for i in range(sides)]
        raw.polygon(polygon,fill=fill,outline=edge,width=max(ss,round(1.5*ss)))
        label=str(ord(point.label.upper())-ord('A')+1) if style['label_set']=='numbers' else point.label
        bounds=font.getbbox(label)
        raw.text((cx-(bounds[0]+bounds[2])//2,cy-(bounds[1]+bounds[3])//2),label,font=font,fill=cls.CANDIDATE_TEXT_COLOR)
