from __future__ import annotations

from data.task_spec import TaskFamily, TaskRoute
from data.common.verifiers.routes import VerifierRoute


_VERIFIER = VerifierRoute("data.board_game.core.verifier", "verify_board_game")

FAMILY = TaskFamily(
    name="board_game",
    description="Execute explicit rule sequences on unified numeric-coordinate boards, or search for a complete, independently verifiable optimal action path.",
    tasks=(
        TaskRoute(
            "chess_l_waypoint_search",
            "data.board_game.chess_l_waypoint_search.generator",
            "ChessLWaypointSearchGenerator",
            verifier=_VERIFIER,
            answer_kind="path",
        ),
        TaskRoute(
            "go_max_capture_search",
            "data.board_game.go_max_capture_search.generator",
            "GoMaxCaptureSearchGenerator",
            verifier=_VERIFIER,
            answer_kind="position",
        ),
        TaskRoute(
            "go_rule_simulator",
            "data.board_game.go_rule_simulator.generator",
            "GoRuleSimulatorGenerator",
            verifier=_VERIFIER,
            answer_kind="grid",
        ),
        TaskRoute(
            "gomoku_rule_simulator",
            "data.board_game.gomoku_rule_simulator.generator",
            "GomokuRuleSimulatorGenerator",
            verifier=_VERIFIER,
            answer_kind="grid",
        ),
        TaskRoute(
            "hex_chain_count",
            "data.board_game.hex_chain_count.generator",
            "HexChainCountGenerator",
            verifier=_VERIFIER,
            answer_kind="nonnegative_integer",
        ),
        TaskRoute(
            "lightsout_single_press",
            "data.board_game.lightsout_single_press.generator",
            "LightsOutSinglePressGenerator",
            verifier=_VERIFIER,
            answer_kind="positions",
            position_collection_order="x_then_y",
        ),
        TaskRoute(
            "mancala_distribute_predict",
            "data.board_game.mancala_distribute_predict.generator",
            "MancalaDistributePredictGenerator",
            verifier=_VERIFIER,
            answer_kind="nonnegative_integer_list",
        ),
        TaskRoute(
            "tic_tac_toe_rule_simulator",
            "data.board_game.tic_tac_toe_rule_simulator.generator",
            "TicTacToeRuleSimulatorGenerator",
            verifier=_VERIFIER,
            answer_kind="grid",
        ),
    ),
)

__all__ = ["FAMILY"]
