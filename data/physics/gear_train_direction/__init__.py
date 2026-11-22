TASK_NAME = 'gear_train_direction'
TASK_FAMILY = 'physics'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Numbered gears form a connected train driven at gear 1 whose direction is fixed by the blue curved arrow. Meshing is visible tangency (rims touch); an open belt (two straight gray lines) joins two separated gears. Rules stated executably in the prompt: meshed gears counter-rotate, belted gears co-rotate, and if a closed loop's contacts cannot all be satisfied the whole train jams and nothing turns. Path samples (4-6 gears, up to two non-adjacent belt links) are always consistent; ring samples close a loop of 3-5 mutually meshing equal gears behind an entry path of 0-2 gears, and the loop jams exactly when its mesh count is odd (3 or 5), while an even loop (4) stays consistent - so seeing a loop is never enough and the parity must actually be counted. The answer is the highest-numbered gear's direction as seen by the viewer, or JAM, from the closed vocabulary CW/CCW/JAM. Non-linked rims keep >=16px clearance so mesh-versus-clear is perceptually crisp (section 7.5); radii vary so rotation speeds differ but direction is radius-independent.",
    "design_intent": 'Make direction propagation a physically rendered machine where adjacency must be read from tangency, two link types behave differently, and loop parity has a real mechanical consequence.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
