TASK_NAME = 'spiral_similarity_center'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A white canvas shows two similar triangles — one black and one dark gray — where the gray triangle is the image of the black one under a single spiral similarity: one fixed centre, one common turn (42-138 degrees in either direction), and one common distance fraction (0.34-0.62). Five labeled candidate circles A-E are shown nearby. Identify the unique fixed centre about which all three black vertices turn through one common angle and scale by one common fraction onto their gray partners. Distinct interior angles (at least 13 degrees apart) determine vertex correspondence visually, and every distractor disagrees across vertex pairs by at least 15 degrees of turn or a factor of at least 1.60 in scale while maintaining outline-incidence parity.',
    "design_intent": 'Spiral Similarity Center makes this exact relation carry correctness: the gray triangle is the black one turned through one common angle about a single fixed centre and shrunk by one common fraction toward it, and the required point is that centre — the only point every vertex pair agrees on. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
