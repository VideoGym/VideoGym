from __future__ import annotations

from data.task_spec import TaskFamily, TaskRoute
from data.common.verifiers.routes import VerifierRoute

REARRANGEMENT_VERIFIER = VerifierRoute(
    'data.rearrangement_game.core.verifier',
    'verify_rearrangement',
)

FAMILY = TaskFamily(
    name='rearrangement_game',
    description='Plan object rearrangement and assignment under visible initial states, goals, order, capacity, queue, stack, or scheduling constraints.',
    tasks=(
        TaskRoute('bookshelf_sort_by_height', 'data.rearrangement_game.bookshelf_sort_by_height.generator', 'BookshelfSortByHeightGenerator', verifier=REARRANGEMENT_VERIFIER, answer_kind='typed_text'),
        TaskRoute('bus_stops_dispatch', 'data.rearrangement_game.bus_stops_dispatch.generator', 'BusStopsDispatchGenerator', verifier=REARRANGEMENT_VERIFIER, answer_kind='nonnegative_integer_list'),
        TaskRoute('chess_pawn_swap', 'data.rearrangement_game.chess_pawn_swap.generator', 'ChessPawnSwapGenerator', verifier=REARRANGEMENT_VERIFIER, answer_kind='typed_text'),
        TaskRoute('color_group_with_locks', 'data.rearrangement_game.color_group_with_locks.generator', 'ColorGroupWithLocksGenerator', verifier=REARRANGEMENT_VERIFIER, answer_kind='nonnegative_integer_list'),
        TaskRoute('container_ship_loading_order', 'data.rearrangement_game.container_ship_loading_order.generator', 'ContainerShipLoadingOrderGenerator', verifier=REARRANGEMENT_VERIFIER, answer_kind='typed_text'),
        TaskRoute('crane_block_stack', 'data.rearrangement_game.crane_block_stack.generator', 'CraneBlockStackGenerator', verifier=REARRANGEMENT_VERIFIER, answer_kind='typed_text'),
        TaskRoute('dock_yard_crane_swap', 'data.rearrangement_game.dock_yard_crane_swap.generator', 'DockYardCraneSwapGenerator', verifier=REARRANGEMENT_VERIFIER, answer_kind='typed_text'),
        TaskRoute('filing_cabinet_alphabet_lock', 'data.rearrangement_game.filing_cabinet_alphabet_lock.generator', 'FilingCabinetAlphabetLockGenerator', verifier=REARRANGEMENT_VERIFIER, answer_kind='typed_text'),
        TaskRoute('jumping_frog_swap', 'data.rearrangement_game.jumping_frog_swap.generator', 'JumpingFrogSwapGenerator', verifier=REARRANGEMENT_VERIFIER, answer_kind='nonnegative_integer_list'),
        TaskRoute('lever_crane_swap', 'data.rearrangement_game.lever_crane_swap.generator', 'LeverCraneSwapGenerator', verifier=REARRANGEMENT_VERIFIER, answer_kind='typed_text'),
        TaskRoute('maze_rush_hour', 'data.rearrangement_game.maze_rush_hour.generator', 'MazeRushHourGenerator', verifier=REARRANGEMENT_VERIFIER, answer_kind='typed_text'),
        TaskRoute('meal_train_pass_serve', 'data.rearrangement_game.meal_train_pass_serve.generator', 'MealTrainPassServeGenerator', verifier=REARRANGEMENT_VERIFIER, answer_kind='nonnegative_integer_list'),
        TaskRoute('office_inbox_priority_pull', 'data.rearrangement_game.office_inbox_priority_pull.generator', 'OfficeInboxPriorityPullGenerator', verifier=REARRANGEMENT_VERIFIER, answer_kind='typed_text'),
        TaskRoute('pancake_flip', 'data.rearrangement_game.pancake_flip.generator', 'PancakeFlipGenerator', verifier=REARRANGEMENT_VERIFIER, answer_kind='nonnegative_integer_list'),
        TaskRoute('parking_2d_n_by_n', 'data.rearrangement_game.parking_2d_n_by_n.generator', 'Parking2DNByNGenerator', verifier=REARRANGEMENT_VERIFIER, answer_kind='typed_text'),
        TaskRoute('parking_lot_swap_minimal_moves', 'data.rearrangement_game.parking_lot_swap_minimal_moves.generator', 'ParkingLotSwapMinimalMovesGenerator', verifier=REARRANGEMENT_VERIFIER, answer_kind='nonnegative_integer_list'),
        TaskRoute('pipeline_hold_cells_3', 'data.rearrangement_game.pipeline_hold_cells_3.generator', 'PipelineHoldCells3Generator', verifier=REARRANGEMENT_VERIFIER, answer_kind='typed_text'),
        TaskRoute('sliding_tile_15', 'data.rearrangement_game.sliding_tile_15.generator', 'FifteenPuzzleBoardStateGenerator', verifier=REARRANGEMENT_VERIFIER, answer_kind='path'),
        TaskRoute('stack_reversal_minimal_moves', 'data.rearrangement_game.stack_reversal_minimal_moves.generator', 'StackReversalMinimalMovesGenerator', verifier=REARRANGEMENT_VERIFIER, answer_kind='typed_text'),
        TaskRoute('token_trade_matrix', 'data.rearrangement_game.token_trade_matrix.generator', 'TokenTradeMatrixGenerator', verifier=REARRANGEMENT_VERIFIER, answer_kind='typed_text'),
        TaskRoute('tower_of_hanoi', 'data.rearrangement_game.tower_of_hanoi.generator', 'TowerOfHanoiGenerator', verifier=REARRANGEMENT_VERIFIER, answer_kind='typed_text'),
        TaskRoute('train_couple_swap', 'data.rearrangement_game.train_couple_swap.generator', 'TrainCoupleSwapGenerator', verifier=REARRANGEMENT_VERIFIER, answer_kind='typed_text'),
        TaskRoute('vending_dispense_fifo', 'data.rearrangement_game.vending_dispense_fifo.generator', 'VendingDispenseFifoGenerator', verifier=REARRANGEMENT_VERIFIER, answer_kind='typed_text'),
    ),
)

__all__ = ["FAMILY"]
