"""Six source-equivalent authored scenes for Vision Prompt Paraphrase.

Drawing substitutions are scoped to the task being generated or observed. The
same functions serve source-derived verifier templates; neither solvers nor
candidate media are changed by this context.
"""
from contextlib import ExitStack, contextmanager
import math
import re
import numpy as np
from unittest.mock import patch

from PIL import Image, ImageDraw

from data.common.render_primitives import new_scaled_canvas, finish_scaled_canvas
from data.common.pipeline.media import load_font

NEW_DESIGNS = {
    'grid_shift': ('token_switchyard', 'Re-embed the row-column grid as a diamond transfer lattice with labelled oblique axes, moving the arrow and coloured ID modules into those coordinates.', 'The 16 addressed sockets retain the same rows, columns, token IDs, cardinal translation and boundary deletion.'),
    'conveyor_belt_path': ('pneumatic_conveyors', 'Replace tiled corridors by a pipe-junction network; forced conveyor cells become directional pneumatic valves.', 'Each original walkable cell is one junction; every original orthogonal legal move is one pipe. Valve directions, blocked cells, start and goal are unchanged.'),
    'pancake_flip': ('flip_foundry', 'Re-express the food stacks as horizontal beam magazines, with the working magazine below its target and leftmost groups reversing through rigid-body half turns.', 'Beam lengths identify exactly the original pancake sizes; left-to-right order maps original top-to-bottom order, and leftmost-k reversal maps exactly the original top-k flip.'),
    'linear_sequence_next': ('paper_sequence', 'Lay out the four observations left-to-right as cream paper punchcards, replacing filled squares with addressed apertures and coloured diamond inserts.', 'The same four ordered observations, labelled six-by-six coordinates, visible patterns and uniquely inferred repeated transformation are preserved.'),
    'sokoban': ('cargo_docks', 'Replace the warehouse tiles with connected docking platforms over water, a cargo container, and a mobile tug.', 'Dock adjacency is exactly the original floor graph; obstructed cells, target, player, box and legal shortest push sequence are unchanged.'),
    'yin_yang_completion': ('ceramic_mosaic', 'Recast the board as an oblique ceramic mosaic: horizontal rows and slanted columns hold black or white inlays, with coral corners identifying fixed tiles.', 'One tile per original cell; edge adjacency, monochrome 2 by 2 exclusions, fixed values and forced deduction waves are unchanged.'),
}


def rewrite_new_prompt(text, profile):
    """Replace the image description and coordinate appearance throughout."""
    from data.common.grid_coordinates import GRID_XY_CONVENTION_SENTENCE
    head, sep, tail = text.partition('\n\nRule:')
    if not sep:
        raise ValueError('a sectioned task prompt is required')
    dimension = re.search(r"(\d+)(?:x| by )(\d+)", head)
    size = f"{dimension[1]} by {dimension[2]}" if dimension else "square"
    descriptions = {
        'token_switchyard': 'A diamond transfer lattice contains four rows and four columns of white sockets. R1–R4 label rows along the upper-left boundary; C1–C4 label columns along the upper-right boundary. Increasing the row moves down-left; increasing the column moves down-right. An empty socket is white. Each coloured module bears its fixed integer ID from 1 to 6. The gold arrow is drawn in these oblique coordinates and its numeral 1 specifies one step along the indicated row or column axis.',
        'pneumatic_conveyors': f'A {size} addressed pipe network is drawn as a clean transport map on ivory paper, with row numbers down the left and column numbers across the top. Pale circular junctions are floor cells and pipes connect exactly legal orthogonal neighbours. A missing junction is a wall cell. Green S and blue G mark the start and goal; the orange traveller starts at S. Gold circular valves bearing dark chevrons force movement in the chevrons’ direction.',
        'flip_foundry': 'Two horizontal timber-framed magazines are arranged one above the other. The lower magazine is the working state and the upper magazine is the target. Each upright coloured beam has a persistent length and colour identifying that beam. Read the order in each magazine from left to right. A blue chevron points upward from the working magazine to the target. Rails and rivets are visual furniture and add no rule.',
        'paper_sequence': 'Four numbered cream paper punchcards arranged from left to right show successive six-by-six arrays of circular apertures, with rows and columns numbered 1–6. Cards 1–3 contain coloured diamond inserts at their occupied positions; card 4 contains a question mark over an empty array. Empty apertures and inserted diamonds represent the two logical cell states. Paper perforations and ruled guides add no extra information.',
        'cargo_docks': f'A {size} coordinate map is drawn as docking platforms over pale turquoise water, with row and column labels along the left and top. Platforms are floor cells; water gaps are blocked wall cells. Footbridges join exactly the orthogonally adjacent floor cells. A yellow tug with a dark windscreen is the agent, the brown ribbed shipping container is the pushable box, and the green cross marks the destination platform. Water-wave decoration has no task meaning.',
        'ceramic_mosaic': 'A slightly slanted ceramic grid has horizontal rows numbered down the left and columns numbered across the top. Each recess is one grid cell and holds at most one black or white tile. Coral corners identify the fixed given tiles; other recesses are initially empty. Thin grout lines align immediate row and column neighbours. The small slant changes the drawing only: edge adjacency and every local 2 by 2 block use the ordinary two adjacent rows and two adjacent columns.',
    }
    substitutions = {
        'token_switchyard': [('numbered tokens', 'numbered transfer modules'), ('numbered token', 'numbered transfer module'), ('token', 'module'), ('right by 1', 'toward the next larger column label (down-right)'), ('left by 1', 'toward the next smaller column label (up-left)'), ('up by 1', 'toward the next smaller row label (up-right)'), ('down by 1', 'toward the next larger row label (down-left)')],
        'pneumatic_conveyors': [('non-conveyor cell', 'junction without a valve'), ('conveyor belt', 'directional valve'), ('conveyor', 'directional valve'), ('belt', 'valve'), ('wall', 'missing junction')],
        'flip_foundry': [('pancakes', 'rigid beams'), ('pancake', 'rigid beam'), ('plates', 'magazines'), ('plate', 'magazine'), ('stack height', 'number of beams'), ('top K', 'leftmost K'), ('top prefix', 'leftmost group'), ('below the prefix', 'to the right of the group'), ('width', 'length'), ('stack', 'magazine')],
        'paper_sequence': [('filled cells', 'occupied apertures'), ('filled cell', 'occupied aperture'), ('cards', 'punchcards'), ('card', 'punchcard'), ('cells filled at the same cell color', 'apertures occupied by diamond inserts of the same colour')],
        'cargo_docks': [('wooden crate', 'shipping container'), ('crate', 'shipping container'), ('box', 'shipping container'), ('agent', 'tug'), ('player', 'tug'), ('walls', 'water gaps'), ('wall', 'water gap')],
        'ceramic_mosaic': [('black and white discs', 'black and white ceramic tiles'), ('ringed', 'corner-marked'), ('stones', 'tiles'), ('stone', 'tile'), ('parchment cells', 'mosaic recesses'), ('fully stoned board', 'completed mosaic')],
    }
    for before, after in substitutions[profile]:
        tail = re.sub(r'\b' + re.escape(before) + r'\b',
                      lambda match: after[0].upper()+after[1:] if match[0][0].isupper() else after,
                      tail, flags=re.IGNORECASE)
    if profile == 'flip_foundry':
        _, shared_separator, shared = tail.partition('\n\nBegin with')
        if not shared_separator:
            raise ValueError('the source delivery constraints are required')
        tail = (
            ' Each move chooses an integer K from 2 to the number of beams and reverses the '
            'leftmost K beams in the working magazine by a rigid half turn. Beams to the right '
            'of that group stay in place. Track each beam by its persistent length and colour.\n\n'
            'Required Output: Show a complete legal sequence of group reversals that makes the '
            'lower working magazine match the upper target. Withdraw the chosen leftmost group from '
            'the remaining beams, turn it continuously as one rigid assembly through a half turn, '
            'then return and commit its complete reversed order together. '
            'All beams outside that group remain fixed.' + shared_separator + shared
        )
    description = descriptions[profile]
    if GRID_XY_CONVENTION_SENTENCE in head:
        description += ' ' + GRID_XY_CONVENTION_SENTENCE
    return 'Input Image: ' + description + sep + tail


class QuarterTurnChart:
    """Exact spatial bijection from a vertical stack to a horizontal magazine."""
    def apply(self, image):
        return np.rot90(np.asarray(image), 1).copy()

    def undo(self, image):
        return np.rot90(np.asarray(image), -1).copy()


def scene_chart(plan):
    return QuarterTurnChart() if plan.get('profile') in {'flip_foundry', 'hanoi_crates'} else None


def _font(size, factor=4):
    return load_font(max(7, int(size)) * factor)



def _transfer_center(row,column):
    return 256+(column-row)*47,166+(column+row)*34


def _grid_shift(self, task, matrix, *, status):
    from data.next_state_prediction.grid_shift.generator import _PALETTE
    image, draw = new_scaled_canvas(512,512,4,(247,251,253))
    draw.text((256,43),'TRANSFER ARRAY',font=_font(17),fill=(47,78,96),anchor='mm')
    draw.line((60,65,452,65),fill=(179,209,222),width=1)
    draw.text((256,448),'R = ROW   /   C = COLUMN',font=_font(11),fill=(82,116,132),anchor='mm')
    for r in range(4):
        for c in range(4):
            x,y=_transfer_center(r,c)
            for dr,dc in ((1,0),(0,1)):
                if r+dr<4 and c+dc<4:
                    xx,yy=_transfer_center(r+dr,c+dc)
                    draw.line((x,y,xx,yy),fill=(140,185,202),width=3)
    for r in range(4):
        for c in range(4):
            x,y=_transfer_center(r,c)
            draw.polygon([(x,y-25),(x+34,y),(x,y+25),(x-34,y)],fill=(255,255,255),outline=(120,165,185),width=1.5)
            value=matrix[r][c]
            if value is not None:
                color=_PALETTE[int(value)];radius=18
                draw.polygon([(x-radius,y-radius*.6),(x-radius*.6,y-radius),(x+radius*.6,y-radius),(x+radius,y-radius*.6),(x+radius,y+radius*.6),(x+radius*.6,y+radius),(x-radius*.6,y+radius),(x-radius,y+radius*.6)],fill=color,outline=(38,71,88),width=1.5)
                draw.text((x,y),str(value),font=_font(14),fill=(255,255,255),anchor='mm')
    for i in range(4):
        x,y=_transfer_center(i,0)
        draw.text((x-42,y),f'R{i+1}',font=_font(11),fill=(48,88,109),anchor='mm')
        x,y=_transfer_center(0,i)
        draw.text((x+33,y-24),f'C{i+1}',font=_font(11),fill=(48,88,109),anchor='mm')
    dr,dc=task['shift'];dx,dy=(dc-dr)*47,(dc+dr)*34
    norm=math.hypot(dx,dy);ux,uy=dx/norm,dy/norm
    cx,cy=256,108
    draw.rounded_rectangle((211,80,301,137),radius=7,fill=(255,231,176),outline=(220,175,90),width=1)
    tail=(cx-ux*19,cy-uy*19);tip=(cx+ux*19,cy+uy*19)
    draw.line((tail,tip),fill=(110,85,38),width=3)
    draw.polygon([tip,(tip[0]-ux*10-uy*5,tip[1]-uy*10+ux*5),(tip[0]-ux*10+uy*5,tip[1]-uy*10-ux*5)],fill=(110,85,38))
    draw.text((319,108),'1',font=_font(17),fill=(110,85,38),anchor='mm')
    return finish_scaled_canvas(image,(512,512))


def _pipes(self, draw, *, wall_set=()):
    walls=set(wall_set); cell=self.cell_size
    width,height=self._canvas_size()
    draw.rectangle((0,0,width,height),fill=(253,250,241))
    for r in range(self.size):
        for c in range(self.size):
            if (r,c) in walls:continue
            x,y=self._cell_center((r,c))
            for dr,dc in ((0,1),(1,0)):
                if r+dr<self.size and c+dc<self.size and (r+dr,c+dc) not in walls:
                    xx,yy=self._cell_center((r+dr,c+dc))
                    draw.line((x,y,xx,yy),fill=(64,137,154),width=cell*.25)
                    draw.line((x,y,xx,yy),fill=(189,226,224),width=cell*.1)
            draw.ellipse((x-cell*.3,y-cell*.3,x+cell*.3,y+cell*.3),fill=(255,255,255),outline=(64,137,154),width=3)
    font=self._load_font(max(12,cell//4))
    for i in range(self.size):
        draw.text((self.margin-cell*.32,self.margin+(i+.5)*cell),str(i+1),font=font,fill=(67,91,94),anchor='mm')
        draw.text((self.margin+(i+.5)*cell,self.margin-cell*.32),str(i+1),font=font,fill=(67,91,94),anchor='mm')


def _valve(self,draw,cell,*,direction):
    x,y=self._cell_center(cell); s=self.cell_size
    dr,dc={'right':(0,1),'left':(0,-1),'up':(-1,0),'down':(1,0)}[direction]
    draw.ellipse((x-s*.33,y-s*.33,x+s*.33,y+s*.33),fill=(238,184,67),outline=(120,82,32),width=2)
    for offset in (-.12,.1):
        cx,cy=x+dc*s*offset,y+dr*s*offset
        pts=[(cx-dc*s*.09-dr*s*.12,cy-dr*s*.09+dc*s*.12),(cx+dc*s*.07,cy+dr*s*.07),(cx-dc*s*.09+dr*s*.12,cy-dr*s*.09-dc*s*.12)]
        draw.line(pts,fill=(58,72,74),width=3)


def _tile_center(row,column):
    return 66+column*58+row*16,84+row*64


def _ceramic_tile(x,y,half_width,half_height):
    shear=half_height*.25
    return [(x-half_width-shear,y-half_height),(x+half_width-shear,y-half_height),
            (x+half_width+shear,y+half_height),(x-half_width+shear,y+half_height)]


def _yin_yang(self,committed,given):
    n=self.size
    image,draw=new_scaled_canvas(512,512,4,(253,249,241))
    for r in range(n):
        for c in range(n):
            x,y=_tile_center(r,c)
            if c+1<n:
                xx,yy=_tile_center(r,c+1);draw.line((x,y,xx,yy),fill=(214,191,165),width=1.5)
            if r+1<n:
                xx,yy=_tile_center(r+1,c);draw.line((x,y,xx,yy),fill=(214,191,165),width=1.5)
    for r in range(n):
        for c in range(n):
            x,y=_tile_center(r,c)
            draw.polygon(_ceramic_tile(x+2,y+3,24,26),fill=(226,211,192))
            draw.polygon(_ceramic_tile(x,y,24,26),fill=(243,230,211),outline=(180,153,126),width=1.5)
            value=committed.get((r,c))
            if value is not None:
                tile=_ceramic_tile(x,y,20,22)
                draw.polygon(tile,fill=(32,36,44) if value else (255,255,255),outline=(141,133,121),width=1.3)
                if (r,c) in given:
                    outer=_ceramic_tile(x,y,27,29)
                    for i,corner in enumerate(outer):
                        previous=outer[(i-1)%4];following=outer[(i+1)%4]
                        arm=lambda point:(corner[0]+(point[0]-corner[0])*.19,corner[1]+(point[1]-corner[1])*.19)
                        draw.line([arm(previous),corner,arm(following)],fill=(220,90,70),width=2.5)
    for r in range(n):
        x,y=_tile_center(r,0)
        draw.text((x-44,y),str(r+1),font=_font(12),fill=(117,96,79),anchor='mm')
    for c in range(n):
        x,y=_tile_center(0,c)
        draw.text((x-9,y-41),str(c+1),font=_font(12),fill=(117,96,79),anchor='mm')
    return finish_scaled_canvas(image,(512,512))


def _paper_card_boxes(arrangement,total):
    if arrangement!='grid4' or total!=4:
        raise ValueError('the paper sequence has exactly four indexed observations')
    return [(9+126*i,174,125+126*i,338) for i in range(4)]


def _paper_frame(draw,box):
    from data.in_context_learning.linear_sequence_next.generator import _grid_geometry
    left,top,cell=_grid_geometry(box)
    draw.rounded_rectangle(box,radius=3,fill=(255,249,232),outline=(183,164,130),width=1.2)
    for r in range(6):
        y=top+(r+.5)*cell
        draw.line((left,y,left+cell*6,y),fill=(223,210,180),width=.7)
    for r in range(6):
        for c in range(6):
            x,y=left+(c+.5)*cell,top+(r+.5)*cell
            draw.ellipse((x-cell*.43,y-cell*.43,x+cell*.43,y+cell*.43),fill=(255,255,255),outline=(176,164,140),width=.8)
    for i in range(6):
        draw.text((left+(i+.5)*cell,top-7),str(i+1),font=_font(9),fill=(113,101,79),anchor='mm')
        draw.text((left-8,top+(i+.5)*cell),str(i+1),font=_font(9),fill=(113,101,79),anchor='mm')


def _paper_cells(draw,box,item,position):
    from data.in_context_learning.linear_sequence_next.generator import _grid_geometry
    _paper_frame(draw,box)
    left,top,cell=_grid_geometry(box)
    color=tuple(item['color'])
    for r,c in item['cells']:
        x,y=left+(c+.5)*cell,top+(r+.5)*cell; radius=cell*.42
        draw.polygon([(x,y-radius),(x+radius,y),(x,y+radius),(x-radius,y)],fill=color,outline=(230,244,238),width=1)


def _cargo(self,board,player,box,target,*,player_center_override=None,box_center_override=None):
    cell,margin=self.cell_size,self.margin
    width,height=self.width*cell+2*margin,self.height*cell+2*margin
    image,draw=new_scaled_canvas(width,height,4,(214,241,245))
    for y in range(12,height,22):
        for x in range(10,width,44):draw.arc((x,y,x+23,y+8),0,180,fill=(154,205,213),width=1)
    for r in range(self.height):
        for c in range(self.width):
            if board[r][c]=='#':continue
            x,y=margin+(c+.5)*cell,margin+(r+.5)*cell
            for dr,dc in ((0,1),(1,0)):
                if r+dr<self.height and c+dc<self.width and board[r+dr][c+dc]!='#':
                    draw.line((x,y,x+dc*cell,y+dr*cell),fill=(213,186,147),width=cell*.5)
            d=cell*.39
            draw.polygon([(x-d,y-d*.75),(x+d*.75,y-d),(x+d,y+d*.75),(x-d*.75,y+d)],fill=(255,242,206),outline=(180,149,113),width=2)
    for i in range(self.height):draw.text((margin*.5,margin+(i+.5)*cell),str(i+1),font=_font(max(13,cell//3)),fill=(51,105,119),anchor='mm')
    for i in range(self.width):draw.text((margin+(i+.5)*cell,margin*.5),str(i+1),font=_font(max(13,cell//3)),fill=(51,105,119),anchor='mm')
    tx,ty=margin+(target[1]+.5)*cell,margin+(target[0]+.5)*cell
    d=cell*.31
    draw.rectangle((tx-d,ty-d,tx+d,ty+d),outline=(46,160,67),width=3)
    draw.line((tx-d,ty-d,tx+d,ty+d),fill=(46,160,67),width=2)
    draw.line((tx-d,ty+d,tx+d,ty-d),fill=(46,160,67),width=2)
    bx,by=box_center_override or (margin+(box[1]+.5)*cell,margin+(box[0]+.5)*cell)
    d=(cell-14)/2
    draw.rectangle((bx-d,by-d,bx+d,by+d),fill=(160,113,74),outline=(102,68,43),width=2)
    for dx in (-.5,0,.5):draw.line((bx+dx*d,by-d+3,bx+dx*d,by+d-3),fill=(102,68,43),width=2)
    px,py=player_center_override or (margin+(player[1]+.5)*cell,margin+(player[0]+.5)*cell)
    d=(cell-20)/2
    draw.rounded_rectangle((px-d,py-d,px+d,py+d),radius=3,fill=(255,214,10),outline=(176,135,0),width=2)
    draw.rectangle((px-d*.5,py-d*.55,px+d*.5,py-d*.1),fill=(90,102,100))
    return finish_scaled_canvas(image,(width,height))


def _beam_sprite(image,center,width,height,color,angle,factor):
    padding=4*factor
    sprite=Image.new('RGBA',(round(width*factor)+2*padding,round(height*factor)+2*padding),(0,0,0,0))
    draw=ImageDraw.Draw(sprite)
    x0=y0=padding;x1=x0+width*factor;y1=y0+height*factor
    draw.rectangle((x0,y0,x1,y1),fill=color,outline=(42,54,64),width=2*factor)
    for x in (x0+5*factor,x1-5*factor):
        for y in (y0+4.5*factor,y1-4.5*factor):
            draw.ellipse((x-1.5*factor,y-1.5*factor,x+1.5*factor,y+1.5*factor),fill=(230,236,231))
    rotated=sprite.rotate(angle,resample=Image.Resampling.BICUBIC,expand=True)
    image.paste(rotated,(round(center[0]*factor-rotated.width/2),round(center[1]*factor-rotated.height/2)),rotated)


def _magazine_geometry(layout,count,panel_width):
    """Fit every legal prefix half turn from visible count and dimensions only."""
    height=layout['item_h'];width=layout['max_item_w']
    # A whole stack's furthest possible corner bounds every prefix and order.
    radius=math.hypot(((count-1)*(height+1)+height)/2,width/2)
    scale=min(1,(panel_width/2-12)/radius)
    return height*scale,(height+1)*scale,width*scale


def _magazine_pose(depth,count,k,progress,*,cx,floor,height,pitch,max_width):
    y=floor-depth*pitch-height/2
    if not k or depth<count-k:return (cx,y),0.0
    centre=floor-((count-k+count-1)/2)*pitch-height/2
    radius=math.hypot(((k-1)*pitch+height)/2,max_width/2)
    support_top=floor-(count-k-1)*pitch-height if count>k else floor
    lift=max(0,centre+radius-support_top+4)
    if progress<=.2:
        translation=lift*progress/.2;angle=0.0
    elif progress<.8:
        translation=lift;angle=math.pi*(progress-.2)/.6
    else:
        translation=lift*(1-(progress-.8)/.2);angle=math.pi
    radial=y-centre
    return (cx+radial*math.sin(angle),centre+radial*math.cos(angle)-translation),angle


def _foundry(self,state,panel_w,panel_h,*,flipping=None):
    n=len(state);layout=self._layout(panel_w,panel_h,n)
    image,draw=self._panel_canvas(panel_w,panel_h,(255,250,236))
    cx=layout['cx'];floor=layout['slot_floor_y']
    item_h,pitch,max_width=_magazine_geometry(layout,n,panel_w)
    draw.rectangle((5,10,panel_w-5,panel_h-8),outline=(169,136,95),width=3)
    for x in (10,panel_w-10):
        draw.line((x,15,x,panel_h-15),fill=(210,170,108),width=4)
        for y in range(20,panel_h-15,20):draw.line((x-3,y,x+3,y+6),fill=(157,118,73),width=2)
    draw.rectangle((8,floor+4,panel_w-8,floor+12),fill=(191,155,111))
    k=int(flipping['k']) if flipping else 0
    progress=float(flipping['progress']) if flipping else 0
    for depth,pid in enumerate(state):
        centre,angle=_magazine_pose(depth,n,k,progress,cx=cx,floor=floor,height=item_h,pitch=pitch,max_width=max_width)
        _beam_sprite(image,centre,self._pancake_width(pid,n,max_width),item_h,self._pancake_color(pid,n),math.degrees(angle),draw.factor)
    return self._panel_at_scene_scale(image,panel_w,panel_h)


@contextmanager
def new_scene_context(sample,plan):
    """Use authored task renderers for both generation and template observation."""
    task=sample.task_name
    if task not in NEW_DESIGNS or plan.get('profile')!=NEW_DESIGNS[task][0]:
        yield
        return
    with ExitStack() as stack:
        if task=='grid_shift':
            stack.enter_context(patch('data.next_state_prediction.grid_shift.generator.GridShiftPattern._render_state',_grid_shift))
        elif task=='conveyor_belt_path':
            from data.spatial_navigation.conveyor_belt_path.generator import ConveyorBeltPathGenerator
            stack.enter_context(patch.object(ConveyorBeltPathGenerator,'_draw_grid_and_labels',_pipes))
            stack.enter_context(patch.object(ConveyorBeltPathGenerator,'_draw_conveyor',_valve))
        elif task=='pancake_flip':
            stack.enter_context(patch('data.rearrangement_game.pancake_flip.generator.PancakeFlipGenerator._render_panel',_foundry))
        elif task=='linear_sequence_next':
            stack.enter_context(patch('data.in_context_learning.core.sequence_cards.card_bboxes',_paper_card_boxes))
            stack.enter_context(patch('data.in_context_learning.core.sequence_cards.BACKGROUND',(252,250,244)))
            stack.enter_context(patch('data.in_context_learning.core.sequence_cards.CARD_EDGE',(183,164,130)))
            stack.enter_context(patch('data.in_context_learning.linear_sequence_next.generator._paint_grid_frame',_paper_frame))
            stack.enter_context(patch('data.in_context_learning.linear_sequence_next.generator._paint_cells',_paper_cells))
        elif task=='sokoban':
            stack.enter_context(patch('data.dynamics_game.sokoban.generator.SokobanPlanGenerator._render_board',_cargo))
        elif task=='yin_yang_completion':
            stack.enter_context(patch('data.constraint_grid.yin_yang_completion.generator.YinYangCompletionGenerator._render',_yin_yang))
        yield
