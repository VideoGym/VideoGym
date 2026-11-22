from __future__ import annotations

from data.task_spec import TaskFamily, TaskRoute
from data.common.verifiers.routes import VerifierRoute

FAMILY = TaskFamily(
    name='spatial_navigation',
    description='Plan agents, objects, mechanisms, resources, and legal navigation through discrete grids, walled corridors, hex, toroidal, layered, or pipe channels.',
    tasks=(
        TaskRoute('agent_courier_relay', 'data.spatial_navigation.agent_courier_relay.generator', 'AgentCourierRelayGenerator', verifier=VerifierRoute('data.spatial_navigation.core.verifier', 'verify_spatial_navigation'), answer_kind='named_paths'),
        TaskRoute('bridge_drop_tile', 'data.spatial_navigation.bridge_drop_tile.generator', 'BridgeDropTileGenerator', verifier=VerifierRoute('data.spatial_navigation.core.verifier', 'verify_spatial_navigation'), answer_kind='path'),
        TaskRoute('bus_route_passengers', 'data.spatial_navigation.bus_route_passengers.generator', 'BusRoutePassengersGenerator', verifier=VerifierRoute('data.spatial_navigation.core.verifier', 'verify_spatial_navigation'), answer_kind='path'),
        TaskRoute('carry_capacity_2', 'data.spatial_navigation.carry_capacity_2.generator', 'CarryCapacity2Generator', verifier=VerifierRoute('data.spatial_navigation.core.verifier', 'verify_spatial_navigation'), answer_kind='path'),
        TaskRoute('color_matched_collection', 'data.spatial_navigation.color_matched_collection.generator', 'ColorMatchedCollectionGenerator', verifier=VerifierRoute('data.spatial_navigation.core.verifier', 'verify_spatial_navigation'), answer_kind='path'),
        TaskRoute('conveyor_belt_path', 'data.spatial_navigation.conveyor_belt_path.generator', 'ConveyorBeltPathGenerator', verifier=VerifierRoute('data.spatial_navigation.core.verifier', 'verify_spatial_navigation'), answer_kind='path'),
        TaskRoute('cooperative_open_door', 'data.spatial_navigation.cooperative_open_door.generator', 'CooperativeOpenDoorGenerator', verifier=VerifierRoute('data.spatial_navigation.core.verifier', 'verify_spatial_navigation'), answer_kind='named_paths'),
        TaskRoute('directional_arrow_entry', 'data.spatial_navigation.directional_arrow_entry.generator', 'DirectionalArrowEntryGenerator', verifier=VerifierRoute('data.spatial_navigation.core.verifier', 'verify_spatial_navigation'), answer_kind='path'),
        TaskRoute('falling_block_dodge', 'data.spatial_navigation.falling_block_dodge.generator', 'FallingBlockDodgeGenerator', verifier=VerifierRoute('data.spatial_navigation.core.verifier', 'verify_spatial_navigation'), answer_kind='path'),
        TaskRoute('grid_highest_cost', 'data.spatial_navigation.grid_highest_cost.generator', 'GridHighestCostGenerator', verifier=VerifierRoute('data.spatial_navigation.core.verifier', 'verify_spatial_navigation'), answer_kind='path'),
        TaskRoute('grid_obtaining_award', 'data.spatial_navigation.grid_obtaining_award.generator', 'GridObtainingAwardGenerator', verifier=VerifierRoute('data.spatial_navigation.core.verifier', 'verify_spatial_navigation'), answer_kind='path'),
        TaskRoute('heading_turn_budget_route', 'data.spatial_navigation.heading_turn_budget_route.generator', 'HeadingTurnBudgetRouteGenerator', verifier=VerifierRoute('data.spatial_navigation.core.verifier', 'verify_spatial_navigation'), answer_kind='path'),
        TaskRoute('hidden_treasure_dig', 'data.spatial_navigation.hidden_treasure_dig.generator', 'HiddenTreasureDigGenerator', verifier=VerifierRoute('data.spatial_navigation.core.verifier', 'verify_spatial_navigation'), answer_kind='path'),
        TaskRoute('ice_physics_multi_slide', 'data.spatial_navigation.ice_physics_multi_slide.generator', 'IcePhysicsMultiSlideGenerator', verifier=VerifierRoute('data.spatial_navigation.core.verifier', 'verify_spatial_navigation'), answer_kind='path'),
        TaskRoute('lantern_oil_route', 'data.spatial_navigation.lantern_oil_route.generator', 'LanternOilRouteGenerator', verifier=VerifierRoute('data.spatial_navigation.core.verifier', 'verify_spatial_navigation'), answer_kind='path'),
        TaskRoute('lever_unlock_sequence', 'data.spatial_navigation.lever_unlock_sequence.generator', 'LeverUnlockSequenceGenerator', verifier=VerifierRoute('data.spatial_navigation.core.verifier', 'verify_spatial_navigation'), answer_kind='path'),
        TaskRoute('maze_aqueduct_pipes', 'data.spatial_navigation.maze_aqueduct_pipes.generator', 'MazeAqueductPipesGenerator', verifier=VerifierRoute('data.spatial_navigation.core.path_verifier', 'verify_path_trace'), answer_kind='path'),
        TaskRoute('maze_colored_zone', 'data.spatial_navigation.maze_colored_zone.generator', 'MazeColoredZoneGenerator', verifier=VerifierRoute('data.spatial_navigation.core.path_verifier', 'verify_path_trace'), answer_kind='path'),
        TaskRoute('maze_elevation', 'data.spatial_navigation.maze_elevation.generator', 'MazeElevationGenerator', verifier=VerifierRoute('data.spatial_navigation.core.path_verifier', 'verify_path_trace'), answer_kind='path'),
        TaskRoute('maze_hexagon', 'data.spatial_navigation.maze_hexagon.generator', 'MazeHexagonGenerator', verifier=VerifierRoute('data.spatial_navigation.core.path_verifier', 'verify_path_trace'), answer_kind='path'),
        TaskRoute('maze_keys_and_locks', 'data.spatial_navigation.maze_keys_and_locks.generator', 'MazeKeysAndLocksGenerator', verifier=VerifierRoute('data.spatial_navigation.core.path_verifier', 'verify_path_trace'), answer_kind='path'),
        TaskRoute('maze_labyrinth', 'data.spatial_navigation.maze_labyrinth.generator', 'MazeLabyrinthGenerator', verifier=VerifierRoute('data.spatial_navigation.core.path_verifier', 'verify_path_trace'), answer_kind='json_object'),
        TaskRoute('maze_layered', 'data.spatial_navigation.maze_layered.generator', 'MazeLayeredGenerator', verifier=VerifierRoute('data.spatial_navigation.core.path_verifier', 'verify_path_trace'), answer_kind='path'),
        TaskRoute('maze_square', 'data.spatial_navigation.maze_square.generator', 'MazeGenerator', verifier=VerifierRoute('data.spatial_navigation.core.path_verifier', 'verify_path_trace'), answer_kind='path'),
        TaskRoute('maze_toroidal', 'data.spatial_navigation.maze_toroidal.generator', 'MazeToroidalGenerator', verifier=VerifierRoute('data.spatial_navigation.core.path_verifier', 'verify_path_trace'), answer_kind='path'),
        TaskRoute('one_way_portal_chain', 'data.spatial_navigation.one_way_portal_chain.generator', 'OneWayPortalChainGenerator', verifier=VerifierRoute('data.spatial_navigation.core.verifier', 'verify_spatial_navigation'), answer_kind='path'),
        TaskRoute('parcel_delivery_route', 'data.spatial_navigation.parcel_delivery_route.generator', 'ParcelDeliveryRouteGenerator', verifier=VerifierRoute('data.spatial_navigation.core.verifier', 'verify_spatial_navigation'), answer_kind='path'),
        TaskRoute('patrol_avoid_pattern', 'data.spatial_navigation.patrol_avoid_pattern.generator', 'PatrolAvoidPatternGenerator', verifier=VerifierRoute('data.spatial_navigation.core.verifier', 'verify_spatial_navigation'), answer_kind='path'),
        TaskRoute('plant_watering_route', 'data.spatial_navigation.plant_watering_route.generator', 'PlantWateringRouteGenerator', verifier=VerifierRoute('data.spatial_navigation.core.verifier', 'verify_spatial_navigation'), answer_kind='path'),
        TaskRoute('switch_toggle_walls', 'data.spatial_navigation.switch_toggle_walls.generator', 'SwitchToggleWallsGenerator', verifier=VerifierRoute('data.spatial_navigation.core.verifier', 'verify_spatial_navigation'), answer_kind='path'),
        TaskRoute('tide_water_cycle', 'data.spatial_navigation.tide_water_cycle.generator', 'TideWaterCycleGenerator', verifier=VerifierRoute('data.spatial_navigation.core.verifier', 'verify_spatial_navigation'), answer_kind='path'),
    ),
)

__all__ = ["FAMILY"]
