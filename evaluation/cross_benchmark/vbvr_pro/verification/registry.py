"""Map stable engine names to the task-family implementations."""

import importlib

ENGINES = {
    "selection": "evaluation.cross_benchmark.vbvr_pro.verification.annotation.selection",
    "grid": "evaluation.cross_benchmark.vbvr_pro.verification.navigation.grid",
    "scene": "evaluation.cross_benchmark.vbvr_pro.verification.scene.verifier",
    "states": "evaluation.cross_benchmark.vbvr_pro.verification.state_prediction.verifier",
    "puzzles": "evaluation.cross_benchmark.vbvr_pro.verification.rearrangement.puzzles",
    "traffic": "evaluation.cross_benchmark.vbvr_pro.verification.state_prediction.traffic",
    "physics": "evaluation.cross_benchmark.vbvr_pro.verification.physics.verifier",
    "linework": "evaluation.cross_benchmark.vbvr_pro.verification.annotation.linework",
    "counting": "evaluation.cross_benchmark.vbvr_pro.verification.counting.counting",
    "assembly": "evaluation.cross_benchmark.vbvr_pro.verification.assembly.lego",
}


def engines():
    for module in ENGINES.values():
        yield importlib.import_module(module)


def engine_name(module):
    return next((name for name, path in ENGINES.items() if path == module.__name__), module.__name__.rsplit(".", 1)[-1])
