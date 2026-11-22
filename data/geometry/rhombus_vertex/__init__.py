TASK_NAME = 'rhombus_vertex'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "The input is one isosceles triangle whose two equal sides form two sides of a rhombus and whose longer third side is the diagonal joining their far endpoints. Reflecting the apex across that long diagonal gives the unique opposite vertex; joining both diagonal endpoints to the reflected point completes four equal sides with opposite sides parallel. The rhombus interior angle at the diagonal endpoints is sampled across 52-104 degrees (apex angle 76-128 degrees), keeping the diagonal at least 22% longer than the two equal sides so the apex is unambiguously identifiable from the shape.",
    "design_intent": 'The visible isosceles triangle makes apex reflection an exact, inspectable construction, and the sampled angle window means the solver reads the shape rather than recognising one memorised figure. The two added segments expose completion of the rhombus and are not decorative reveal strokes.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
