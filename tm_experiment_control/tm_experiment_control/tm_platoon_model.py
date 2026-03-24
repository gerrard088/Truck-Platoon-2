from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TruckState:
    truck_id: int
    target_lane: str = "center"
    target_lane_ref: Optional[Tuple[int, int, int]] = None
    lane_change_request_ts: Optional[float] = None


@dataclass
class PlatoonState:
    truck_ids: List[int]
    trucks: Dict[int, TruckState] = field(init=False)

    def __post_init__(self):
        self.trucks = {truck_id: TruckState(truck_id=truck_id) for truck_id in self.truck_ids}

    @property
    def order(self) -> List[int]:
        return list(self.truck_ids)

    @property
    def leader_id(self) -> Optional[int]:
        return self.truck_ids[0] if self.truck_ids else None

    def truck(self, truck_id: int) -> TruckState:
        return self.trucks[truck_id]

    def reindex(self, new_order: List[int]):
        self.truck_ids = list(new_order)

    def rotate_leader_to_tail(self):
        if len(self.truck_ids) >= 2:
            self.truck_ids = self.truck_ids[1:] + [self.truck_ids[0]]

    def promote_to_lead(self, truck_id: int):
        if truck_id in self.truck_ids:
            self.truck_ids.remove(truck_id)
            self.truck_ids.insert(0, truck_id)

    def detach_tail(self) -> Optional[int]:
        if not self.truck_ids:
            return None
        return self.truck_ids.pop()

    def attach_tail(self, truck_id: int):
        if truck_id not in self.truck_ids:
            self.truck_ids.append(truck_id)

    def reset_lane_change(self, truck_id: int):
        truck = self.truck(truck_id)
        truck.target_lane = "center"
        truck.target_lane_ref = None
        truck.lane_change_request_ts = None

    def start_lane_change(self, truck_id: int, direction: str, lane_ref, request_ts: float):
        truck = self.truck(truck_id)
        truck.target_lane = direction
        truck.target_lane_ref = lane_ref
        truck.lane_change_request_ts = request_ts

    def reset_all_lane_changes(self):
        for truck_id in self.truck_ids:
            self.reset_lane_change(truck_id)
