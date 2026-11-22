TASK_NAME = 'segment_trisection'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A white canvas shows a black line segment with a solid black disc at one endpoint and a small hollow circle at the other, plus five labeled candidate circles A-E spaced out along the segment. The required relation is: split the segment into three equal parts and identify the candidate at the division point one third of the way from the solid black endpoint towards the hollow endpoint. Uniqueness: the solid disc names which endpoint the third is measured from, so the other trisection point is not a second valid answer and the option row is free to cover the whole segment. Presentation parity: every marker sits on the drawn segment, so "which marker is on the line?" carries no signal, and how many markers lie between the answer and the solid disc is drawn uniformly rather than inherited from a row centred on the answer. A distractor is wrong because it divides the segment in some other ratio; each keeps a marker\'s clearance from both endpoint discs.',
    "design_intent": "Segment Trisection makes this exact relation carry correctness: the division point one third of the way from the solid black endpoint towards the hollow endpoint. Naming that endpoint in the drawing is what lets the option row span the whole segment: while the row itself had to say which endpoint was 'nearer', the row could not move without making the second trisection point equally defensible, and the answer sat in the middle of its own cluster. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
