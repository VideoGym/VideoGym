"""Public descriptions of the authored representations, including mapped actions."""
import re

from data.common.visual_style import STYLES
from evaluation.prompt_paraphrase.vision_prompt_paraphrase.new_scenes import NEW_DESIGNS, rewrite_new_prompt


def _replace(text, pairs):
    for before, after in pairs:
        text = text.replace(before, after)
    return text


def prompt_paraphrase_prompt(sample, plan):
    text = sample.prompt
    profile = plan['profile']
    if sample.task_name in NEW_DESIGNS:
        return rewrite_new_prompt(text, profile)
    head, separator, body = text.partition('\n\nRule:')
    if not separator:
        raise ValueError('a sectioned task prompt is required')
    coordinate = head[head.index('Write each grid position'):].split(').', 1)[0] + ').' if 'Write each grid position' in head else ''
    if profile == 'geometry_workbench':
        from evaluation.prompt_paraphrase.vision_prompt_paraphrase.geometry import prompt_paraphrase_description
        text = _replace(text, [
            ('white canvas', 'light diagram surface'),
            ('candidate circles', 'labelled candidate symbols'),
            ('candidate markers', 'labelled candidate symbols'),
            ('remain white', 'retain their unselected appearance'),
            ('straight down', 'perpendicularly'),
        ])
        head, separator, body = text.partition('\n\nRule:')
        style = plan['geometry_style']
        colour = style['given_colour_name']
        head += ' ' + prompt_paraphrase_description(sample.task_name)
        head += f" Candidate symbols are {style['candidate_shape']} markers; their centres are the mathematical points. Given line centrelines are {colour}; cyan is reserved for constructions and red for the selected answer."
        joined = head + separator + body
        joined = _replace(joined, [('black rays', colour+' rays'), ('black line segment', colour+' line segment'),
                                   ('black segment', colour+' segment'), ('black outline', colour+' outline'),
                                   ('fine dark', 'fine '+colour)])
        if style['label_set'] == 'numbers':
            joined = _replace(joined, [('A-E', '1–5'), ('A–E', '1–5'), ('lettered', 'numbered'),
                                       ('Lettered', 'Numbered'), ('letter centres', 'number centres'),
                                       ('letter centre', 'number centre'), ('alphabetical order', 'ascending numerical order')])
        head, separator, body = joined.partition('\n\nRule:')
    elif profile == 'node_maze':
        # Keep source-specific endpoints, key counts and dependency legend.
        head = _replace(head, [
            ('square maze of white walkable cells and dark slate wall cells', 'transit diagram of diamond stations joined by double rails'),
            ('square orthogonal maze', 'transit diagram of diamond stations joined by double rails'),
        ])
        head += ' Each station is one walkable position. A rail joins exactly two side-adjacent positions; blank positions are walls.'
        body = _replace(body, [('side-adjacent white cells', 'stations joined by a rail'),
            ('visited white cells', 'visited stations'), ('dark slate walls', 'blank non-station positions'),
            ('dark slate cells', 'blank non-station positions')])
        if sample.task_name == 'maze_keys_and_locks':
            head = head.replace('transit diagram of diamond stations joined by double rails', 'floorplan of rooms joined by broad passages').replace('station', 'room').replace('rail', 'passage')
            head += ' Coloured keycards unlock matching vault doors; a collected keycard becomes pale and an opened door becomes light green.'
            body = body.replace('stations joined by a rail', 'rooms joined by a passage').replace('station', 'room')
    elif profile == 'portal_gates':
        head = _replace(head, [('three filled-disc cells with thicker rings', 'three filled outward-arrow send emblems'),
            ('three lighter-ring cells', 'three outlined inward-arrow receive emblems'),
            ('grid with row numbers', 'diamond-station network with row numbers')])
        head += ' Thin links join neighbouring floor stations. The labelled send and receive emblems encode the one-way portal pairing.'
        body = body.replace('Do not teleport, duplicate', 'Do not make undeclared teleports, duplicate')
    elif profile == 'distance_beacons':
        head = ('Input Image: A diamond-oriented distance network has pale circular stations for open cells, '
                'green stations for simultaneous sources of distance 0, and dark striped barrier tiles for walls. '
                'Each drawn link is one permitted step between adjacent stations. The two diagonal lattice axes '
                'represent the two axes of the underlying rectangular grid; unlinked near neighbours are not adjacent.')
        body = body.replace('Movement is allowed only one cell up, down, left, or right; diagonal movement is forbidden.',
                            'Movement follows one drawn station link at a time; no unlinked jump is allowed.')
    elif profile == 'disc_lattice':
        head = ('Input Image: An hourglass signal array carries one binary state per indexed position. '
                'Navy hourglasses encode 1 (live), pale hourglasses encode 0 (empty). '
                'The birth and survival sets are specified below. ' + coordinate)
        body = body.replace('ordinary pale cell', 'ordinary pale hourglass')
    elif profile == 'register_chips':
        head = ('Input Image: Five labelled instruments show old register bits Q1–Q3 and fixed inputs X/Y. '
                'The highlighted 0 or 1 in each two-position window is that instrument’s bit. '
                'Instrument positions carry no dependency information; the three D expressions are given below.')
        body = body.replace('table layout', 'instrument layout')
    elif profile == 'height_columns':
        head = ('Input Image: Two circular measuring carousels hold coloured radial rods. '
                'The left carousel is the working arrangement; the right is the target. '
                'A rod’s intrinsic length and colour identify it. Slot 1 points straight up; '
                'subsequent slots run clockwise. The target orders lengths from longest in slot 1 '
                'to shortest in the last clockwise slot. The hub triangle only marks the top.')
        body = _replace(body, [('books', 'rods'), ('book', 'rod'), ('shelf slots', 'carousel slots'),
            ('shelf slot', 'carousel slot'), ('shelves', 'carousels'), ('shelf', 'carousel'),
            ('leftmost slot', 'top slot'), ('from left to right', 'clockwise from the top'),
            ('heights', 'intrinsic lengths'), ('height', 'intrinsic length'), ('slot numbers', 'slot positions')])
        body += (' During a swap the two selected rods travel around the carousel to exchange slots; '
                 'their intrinsic lengths stay constant and each rod keeps an outward-facing orientation. '
                 'The two moving rods take slightly separated continuous tracks so neither disappears behind the other. '
                 'All other rods and the target carousel stay fixed.')
    elif profile == 'hanoi_crates':
        head = ('Input Image: Two racks carry wing-shaped platforms on three horizontal pegs. '
                'The lower rack is the working configuration and the upper rack is the fixed target. '
                'A platform’s vertical span is its size; its colour identifies it. '
                'Peg numbers 1, 2, 3 run from bottom to top. On each peg the leftmost platform is exposed.')
        body = _replace(body, [('topmost disk', 'leftmost exposed platform'), ('top disk', 'leftmost exposed platform'),
            ('on top of another peg', 'onto the exposed end of another peg'),
            ('on top of a strictly smaller disk', 'against the exposed side of a strictly smaller platform'),
            ('Wider disks are larger; pegs are numbered 1, 2, 3 from left to right.',
             'Platforms with a longer vertical span are larger; pegs 1, 2, 3 run from bottom to top.'),
            ('legal top surface', 'legal exposed surface'), ('Lift only', 'Withdraw only'),
            ('disks', 'platforms'), ('disk', 'platform'), ('widths', 'vertical spans')])
    elif profile == 'height_solids':
        head = ('Input Image: Four numbered signal gauges are arranged in one row. Gauges 1–3 carry '
                'the same coloured mast above a baseline, against unit marks 1–8. Gauge 4 has a question mark. '
                'A mast’s highest tip gives its exact height; lateral lobes do not alter that reading.')
        body = _replace(body, [('cards', 'gauges'), ('card', 'gauge'), ('shape', 'signal mast'), ('top edge', 'highest tip')])
    elif profile == 'colour_gems':
        head = ('Input Image: Six numbered stations form an open ring linked by cables. Follow the printed numbers '
                '1 through 6, not an inferred spatial direction. Stations 1–5 contain four-lobed colour emblems; '
                'station 6 contains a question mark. Each emblem’s uniform colour is the sequence value.')
        body = _replace(body, [('cards', 'stations'), ('card', 'station'), ('tiles', 'emblems'), ('tile', 'emblem'),
            ('one solid emblem', 'one four-lobed emblem'), ('Card', 'Station'), ('Tile', 'Emblem')])
    elif profile == 'block_inlays':
        head = _replace(head, [('rounded blue blocks', 'blue bow-tie modules'),
            ('pale wells', 'pale sockets'), ('orange piece', 'orange linked-module piece'),
            ('card to the right', 'tray below the board')])
        head += ' Each bow-tie module occupies exactly one complete cell; the two lobes are one object, not separate blocks.'
        body = _replace(body, [('pale empty well', 'pale empty socket'), ('outlined wells', 'outlined sockets'),
            ('from its card', 'from its supply tray'), ('piece card', 'supply tray'), ('blue blocks', 'blue modules')])
    elif profile == 'crown_queens':
        head = ('Input Image: An indexed array of diamond sockets shows fixed navy compass emblems. '
                'Each compass is one queen; its eight points indicate the usual row, column and diagonal attacks. '
                'Socket shape changes no adjacency or attack relation. ' + coordinate)
        body = re.sub(r'\b(dark|navy) Q marker(s?)\b', lambda m: m[1]+' compass emblem'+m[2], body, flags=re.I)
        body = body.replace('checkered squares', 'diamond sockets').replace('the squares', 'the sockets')
    elif profile == 'tile_go':
        head = ('Input Image: A diamond mesh shows a 7 by 7 Go position with black and white tokens at tile centres. '
                'The top tip is (1, 1). Row numbers increase down the left-sloping axis; column numbers increase '
                'down the right-sloping axis. Write a position as (row, column), both numbered 1–7. '
                'Tiles sharing an edge are neighbours; touching only at a corner gives no adjacency.')
        body = _replace(body, [('horizontal or vertical adjacency', 'shared-edge tile adjacency'),
            ('orthogonally adjacent', 'sharing a tile edge'), ('intersection', 'tile centre'),
            ('wood board', 'diamond board'), ('grid lines', 'tile boundaries')])
    elif profile == 'open_gears':
        head = ('Input Image: A physical assembly of numbered gears has pastel toothed rims and open '
                'centres with rigid fan-shaped spokes. Gears whose rims touch mesh directly. '
                'A belt, when present, loops around its two gears. Gear radii show size and the bold '
                'spokes show orientation. The blue arrow drives gear 1. Read CW and CCW from the '
                'viewer’s side; the numbers name gears and do not specify an execution order.')
        body = _replace(body, [('teeth, shaded sector and spoke pose at rest', 'teeth and spoke pose at rest'),
            ('rotating gear teeth and spokes', 'rotating toothed rims and fan-shaped spokes')])
    elif profile == 'numbered_faces':
        text = _replace(text, [('paper squares', 'rigid assembly panels'), ('paper coloring', 'panel colouring'),
            ('paper colors', 'panel colours'), ('a dice-style pip pattern from 1 to 6 pips', 'an underlined numeral from 1 to 6'),
            ('every pip count', 'every face number'), ('Pips are printed identically', 'Numerals are printed readably'),
            ('in increasing pip order', 'in increasing face-number order'), ('lid pips facing up', 'lid numeral facing up'),
            ('the pip pattern of every square', 'the numeral on every panel'), ('its pip count is the answer', 'its face number is the answer'),
            ('the pip patterns', 'the face numerals'), ('the pips printed on them', 'the numerals printed on them'),
            ('Thin dark edges are the creases', 'Pale panel boundaries are the shared hinges'), ('five unshaded squares', 'five amber panels'), ('rests on the blue base', 'rests on the cyan base'), ('shaded light blue', 'marked cyan'), ('blue-shaded square', 'cyan panel'), ('blue base square', 'cyan base panel')])
        head, separator, body = text.partition('\n\nRule:')
        head += ' Amber assembly panels carry riveted numeral plates; the underline fixes numeral orientation. Only shared hinges connect panels.'
    elif profile in {'radial_graph', 'beacon_graph'}:
        theme = STYLES[plan['skin']]
        head = head.replace('inside the circles', 'inside the '+theme.node_shape)
        if profile == 'beacon_graph':
            head = head.replace('numbered nodes', 'lettered nodes').replace('Numbers inside', 'Letters inside')
        head = head.replace('Each drawn segment is one undirected edge.', 'Each '+theme.edge_name+' is one undirected edge.')
        head += ' Only labelled terminals are vertices; cable crossings create no connection. Terminal positions carry no distance or edge-weight information.'
    else:
        raise ValueError(f'no authored prompt for {profile}')
    return head + separator + body
