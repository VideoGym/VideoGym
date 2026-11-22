"""Class registry for shared next-state prediction tasks."""

from __future__ import annotations

from typing import Dict, Tuple

EXPANSION_CLASS_NAMES: Dict[str, str] = {
    "rule_table_1d_rollout": "RuleTable1DRolloutGenerator",
    "lifelike_rule_rollout": "LifelikeRuleRolloutGenerator",
    "cyclic_threshold_wave": "CyclicThresholdWaveGenerator",
    "excitable_signal_network": "ExcitableSignalNetworkGenerator",
    "alternating_block_rotation": "AlternatingBlockRotationGenerator",
    "xor_parity_lattice": "XorParityLatticeGenerator",
    "sandpile_avalanche": "SandpileAvalancheGenerator",
    "lattice_gas_collision": "LatticeGasCollisionGenerator",
    "majority_consensus": "MajorityConsensusGenerator",
    "forest_fire_barrier": "ForestFireBarrierGenerator",
    "predator_prey_cycle": "PredatorPreyCycleGenerator",
    "turn_write_agent_rollout": "TurnWriteAgentRolloutGenerator",
    "finite_tape_machine_rollout": "FiniteTapeMachineRolloutGenerator",
    "simultaneous_agent_conflict": "SimultaneousAgentConflictGenerator",
    "resource_pick_drop_swarm": "ResourcePickDropSwarmGenerator",
    "diffusion_limited_aggregation": "DiffusionLimitedAggregationGenerator",
    "gui_hierarchical_navigation": "GuiHierarchicalNavigationGenerator",
    "chained_function_pipeline": "ChainedFunctionPipelineGenerator",
    "packet_router_tick": "PacketRouterTickGenerator",
    "rail_signal_interlocking": "RailSignalInterlockingGenerator",
    "elevator_dispatch_tick": "ElevatorDispatchTickGenerator",
    "protocol_handshake_tick": "ProtocolHandshakeTickGenerator",
    "register_machine_tick": "RegisterMachineTickGenerator",
    "clocked_logic_register_tick": "ClockedLogicRegisterTickGenerator",
}

EXPANSION_TASK_NAMES: Tuple[str, ...] = tuple(EXPANSION_CLASS_NAMES)

__all__ = ["EXPANSION_CLASS_NAMES", "EXPANSION_TASK_NAMES"]
