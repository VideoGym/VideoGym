TASK_NAME = 'slice_cross_section'
TASK_FAMILY = 'spatial_3d'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'One solid from a closed family (rectangular box, cube, upright cylinder, upright cone, sphere, upright square pyramid, upright triangular prism) is cut by one flat plane drawn as a thin blue wireframe frame. A 14-row specification pairs each solid with executable cut kinds (horizontal, vertical, tilted, through-the-apex, or off-the-apex) so that exactly one word of the closed vocabulary {circle, ellipse, square, rectangle, triangle, trapezoid} names the true shape of the cut face. Shape-distinguishing parameters are strictly gated: ellipse tilt >= 30 degrees (axis ratio >= 1.155), rectangle side ratio >= 1.30, trapezoid top/bottom ratio in [0.35, 0.70], and apex planes pass strictly through the apex.',
    "design_intent": 'Test 3D cross-section reasoning by predicting the exact planar figure a cut produces, including near-miss pairs such as circle vs. ellipse and square vs. rectangle separated by explicit visual margins.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
