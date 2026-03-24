from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class ScenarioCommandType(Enum):
    NONE = auto()
    START_REORDER = auto()
    START_PROMOTE = auto()


@dataclass(frozen=True)
class ScenarioCommand:
    kind: ScenarioCommandType = ScenarioCommandType.NONE
    direction: str = "right"
    target_truck_id: Optional[int] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlatoonSnapshot:
    platoon_id: str
    order: List[int]
    leader_speed_mps: float
    in_trigger_zone: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorldSnapshot:
    time_s: float
    phase: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseScenarioAgent:
    def reset(self):
        return None

    def decide(self, snapshot: PlatoonSnapshot, world_state: WorldSnapshot) -> ScenarioCommand:
        raise NotImplementedError


class PassiveScenarioAgent(BaseScenarioAgent):
    def decide(self, snapshot: PlatoonSnapshot, world_state: WorldSnapshot) -> ScenarioCommand:
        return ScenarioCommand()


class ScriptedZoneTriggerAgent(BaseScenarioAgent):
    def __init__(self, initial_direction: str = "right"):
        self._next_direction = initial_direction
        self._latched = False

    def reset(self):
        self._latched = False

    def decide(self, snapshot: PlatoonSnapshot, world_state: WorldSnapshot) -> ScenarioCommand:
        if world_state.phase != "IDLE":
            return ScenarioCommand()

        if not snapshot.in_trigger_zone:
            self._latched = False
            return ScenarioCommand()

        if self._latched:
            return ScenarioCommand()

        command = ScenarioCommand(
            kind=ScenarioCommandType.START_REORDER,
            direction=self._next_direction,
            reason=f"scripted trigger at {world_state.time_s:.1f}s",
        )
        self._latched = True
        self._next_direction = "left" if self._next_direction == "right" else "right"
        return command
