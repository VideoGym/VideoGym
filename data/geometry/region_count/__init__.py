TASK_NAME = 'region_count'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Count every connected bounded face whose complete boundary consists of visible circle arcs and exclude the single unbounded outside face; lenses, lobes, overlap subdivisions, and enclosed interiors each count once. Every generated arrangement of two to four circles is connected and in general position: each pair is disjoint or crosses twice, with no tangency, coincidence, containment, shared arc, or triple intersection. Correctness is derived twice per instance: analytically from the planar graph via Euler's relation and independently by raster flood fill; both must agree.",
    "design_intent": 'Region Count makes planar-face topology carry correctness: the solver must read how circle crossings subdivide overlaps into distinct bounded faces and exclude the unbounded outside, on a clean full-canvas arrangement whose only output effect is the single pastel face-fill set.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
