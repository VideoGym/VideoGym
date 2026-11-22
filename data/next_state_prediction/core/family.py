from __future__ import annotations

from data.task_spec import TaskFamily, TaskRoute
from data.common.verifiers.routes import VerifierRoute

FAMILY = TaskFamily(
    name='next_state_prediction',
    description='Parse and predict one or more successor states from the current state and explicit transition rules of fixed rows and columns, grids, signals, or discrete interfaces.',
    tasks=(
        TaskRoute('alternating_block_rotation', 'data.next_state_prediction.alternating_block_rotation.generator', 'AlternatingBlockRotationGenerator', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='json_object'),
        TaskRoute('arc1d_compact_nonzero', 'data.next_state_prediction.arc1d_compact_nonzero.generator', 'Arc1DCompactNonzeroPattern', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='nonnegative_integer_list'),
        TaskRoute('binary_matrix', 'data.next_state_prediction.binary_matrix.generator', 'BinaryMatrixPattern', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='grid'),
        TaskRoute('chained_function_pipeline', 'data.next_state_prediction.chained_function_pipeline.generator', 'ChainedFunctionPipelineGenerator', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='json_object'),
        TaskRoute('clocked_logic_register_tick', 'data.next_state_prediction.clocked_logic_register_tick.generator', 'ClockedLogicRegisterTickGenerator', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='json_object'),
        TaskRoute('connected_components', 'data.next_state_prediction.connected_components.generator', 'ConnectedComponentsPattern', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='nonnegative_integer'),
        TaskRoute('cyclic_threshold_wave', 'data.next_state_prediction.cyclic_threshold_wave.generator', 'CyclicThresholdWaveGenerator', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='json_object'),
        TaskRoute('diffusion_limited_aggregation', 'data.next_state_prediction.diffusion_limited_aggregation.generator', 'DiffusionLimitedAggregationGenerator', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='json_object'),
        TaskRoute('elevator_dispatch_tick', 'data.next_state_prediction.elevator_dispatch_tick.generator', 'ElevatorDispatchTickGenerator', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='json_object'),
        TaskRoute('excitable_signal_network', 'data.next_state_prediction.excitable_signal_network.generator', 'ExcitableSignalNetworkGenerator', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='json_object'),
        TaskRoute('finite_tape_machine_rollout', 'data.next_state_prediction.finite_tape_machine_rollout.generator', 'FiniteTapeMachineRolloutGenerator', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='json_object'),
        TaskRoute('forest_fire_barrier', 'data.next_state_prediction.forest_fire_barrier.generator', 'ForestFireBarrierGenerator', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='json_object'),
        TaskRoute('grid_bfs', 'data.next_state_prediction.grid_bfs.generator', 'GridBFSPattern', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='grid'),
        TaskRoute('grid_component', 'data.next_state_prediction.grid_component.generator', 'GridComponentPattern', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='nonnegative_integer'),
        TaskRoute('grid_shift', 'data.next_state_prediction.grid_shift.generator', 'GridShiftPattern', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='grid'),
        TaskRoute('gui_hierarchical_navigation', 'data.next_state_prediction.gui_hierarchical_navigation.generator', 'GuiHierarchicalNavigationGenerator', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='json_object'),
        TaskRoute('lattice_gas_collision', 'data.next_state_prediction.lattice_gas_collision.generator', 'LatticeGasCollisionGenerator', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='json_object'),
        TaskRoute('lifelike_rule_rollout', 'data.next_state_prediction.lifelike_rule_rollout.generator', 'LifelikeRuleRolloutGenerator', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='json_object'),
        TaskRoute('majority_consensus', 'data.next_state_prediction.majority_consensus.generator', 'MajorityConsensusGenerator', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='json_object'),
        TaskRoute('packet_router_tick', 'data.next_state_prediction.packet_router_tick.generator', 'PacketRouterTickGenerator', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='json_object'),
        TaskRoute('predator_prey_cycle', 'data.next_state_prediction.predator_prey_cycle.generator', 'PredatorPreyCycleGenerator', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='json_object'),
        TaskRoute('protocol_handshake_tick', 'data.next_state_prediction.protocol_handshake_tick.generator', 'ProtocolHandshakeTickGenerator', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='json_object'),
        TaskRoute('rail_signal_interlocking', 'data.next_state_prediction.rail_signal_interlocking.generator', 'RailSignalInterlockingGenerator', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='json_object'),
        TaskRoute('register_machine_tick', 'data.next_state_prediction.register_machine_tick.generator', 'RegisterMachineTickGenerator', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='json_object'),
        TaskRoute('resource_pick_drop_swarm', 'data.next_state_prediction.resource_pick_drop_swarm.generator', 'ResourcePickDropSwarmGenerator', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='json_object'),
        TaskRoute('rotten_oranges', 'data.next_state_prediction.rotten_oranges.generator', 'RottenOrangesPattern', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='integer'),
        TaskRoute('rule_table_1d_rollout', 'data.next_state_prediction.rule_table_1d_rollout.generator', 'RuleTable1DRolloutGenerator', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='json_object'),
        TaskRoute('sandpile_avalanche', 'data.next_state_prediction.sandpile_avalanche.generator', 'SandpileAvalancheGenerator', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='json_object'),
        TaskRoute('simultaneous_agent_conflict', 'data.next_state_prediction.simultaneous_agent_conflict.generator', 'SimultaneousAgentConflictGenerator', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='json_object'),
        TaskRoute('traffic_light_countdown', 'data.next_state_prediction.traffic_light_countdown.generator', 'TrafficLightCountdownPattern', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='json_object'),
        TaskRoute('turn_write_agent_rollout', 'data.next_state_prediction.turn_write_agent_rollout.generator', 'TurnWriteAgentRolloutGenerator', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='json_object'),
        TaskRoute('xor_parity_lattice', 'data.next_state_prediction.xor_parity_lattice.generator', 'XorParityLatticeGenerator', verifier=VerifierRoute('data.next_state_prediction.core.evaluator', 'verify_next_state'), answer_kind='json_object'),
    ),
)

__all__ = ["FAMILY"]
