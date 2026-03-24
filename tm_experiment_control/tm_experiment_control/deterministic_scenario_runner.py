import csv
import os
import sys
import threading
import time
from collections import deque

import carla
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32MultiArray
from std_srvs.srv import SetBool

from .tm_agents import (
    PassiveScenarioAgent,
    PlatoonSnapshot,
    ScenarioCommandType,
    ScriptedZoneTriggerAgent,
    WorldSnapshot,
)
from .tm_platoon_model import PlatoonState

try:
    from agents.navigation.controller import VehiclePIDController
except ModuleNotFoundError:
    carla_pythonapi_dir = os.path.expanduser('~/carla/PythonAPI/carla')
    if os.path.isdir(carla_pythonapi_dir) and carla_pythonapi_dir not in sys.path:
        sys.path.append(carla_pythonapi_dir)
    from agents.navigation.controller import VehiclePIDController


class ManeuverState:
    IDLE = 0
    LEADER_EXITS_LANE = 1
    LEADER_CREATES_GAP = 2
    SUCCESSOR_ENTERS_GAP = 3
    FOLLOWER_ENTERS_GAP = 4
    REORDER_COMPLETE = 5
    COOLDOWN = 6
    LEADER_REENTERS_LANE = 7
    PROMOTE_TARGET_EXITS = 21
    PROMOTE_PLATOON_CREATES_GAP = 22
    PROMOTE_TARGET_REENTERS = 23


class LowLevelController:
    def __init__(self, vehicle, lateral=None, longitudinal=None, max_brake=0.3, max_throttle=0.85):
        lateral = lateral or {'K_P': 1.4, 'K_I': 0.02, 'K_D': 0.12, 'dt': 0.05}
        longitudinal = longitudinal or {'K_P': 0.45, 'K_I': 0.08, 'K_D': 0.04, 'dt': 0.05}
        self.vehicle = vehicle
        self.pid = VehiclePIDController(
            self.vehicle,
            args_lateral=lateral,
            args_longitudinal=longitudinal,
            max_brake=max_brake,
            max_throttle=max_throttle,
        )

    def run_step(self, target_speed_kmh, target_waypoint):
        return self.pid.run_step(target_speed_kmh, target_waypoint)


class LeadNavigator(LowLevelController):
    def __init__(self, vehicle, world_map, target_speed_mps):
        super().__init__(vehicle)
        self.map = world_map
        self.target_speed_mps = float(target_speed_mps)
        self.target_waypoint = None
        self.waypoints_ahead = deque()
        self.reset_waypoints()

    def reset_waypoints(self):
        current_wp = self.map.get_waypoint(self.vehicle.get_location())
        self.target_waypoint = current_wp
        self.waypoints_ahead = deque([current_wp])
        while len(self.waypoints_ahead) < 12:
            if self._append_next_waypoint() is None:
                break

    def _append_next_waypoint(self):
        if not self.waypoints_ahead:
            return None
        current = self.waypoints_ahead[-1]
        next_wpts = current.next(5.0)
        if not next_wpts:
            return None
        driving_yaw = current.transform.rotation.yaw
        next_wp = min(next_wpts, key=lambda x: abs((x.transform.rotation.yaw - driving_yaw) % 360))
        if next_wp.is_junction:
            pairs = next_wp.get_junction().get_waypoints(carla.LaneType.Driving)
            if pairs:
                entries = [pair[0] for pair in pairs] + [pair[1] for pair in pairs]
                next_wp = min(entries, key=lambda x: abs((x.transform.rotation.yaw - driving_yaw) % 360))
        self.waypoints_ahead.append(next_wp)
        return next_wp

    def _prune_passed_waypoints(self):
        passed = 0
        location = self.vehicle.get_location()
        for idx, waypoint in enumerate(self.waypoints_ahead):
            if location.distance(waypoint.transform.location) < 4.0:
                passed = max(passed, idx + 1)
        for _ in range(passed):
            if self.waypoints_ahead:
                self.waypoints_ahead.popleft()
        if not self.waypoints_ahead:
            self.reset_waypoints()
        while len(self.waypoints_ahead) < 12:
            if self._append_next_waypoint() is None:
                break
        self.target_waypoint = self.waypoints_ahead[0]
        return self.target_waypoint

    def step(self, override_waypoint=None, target_speed_mps=None):
        waypoint = override_waypoint or self._prune_passed_waypoints()
        speed_mps = self.target_speed_mps if target_speed_mps is None else float(target_speed_mps)
        return self.run_step(speed_mps * 3.6, waypoint)


class FollowerPathController(LowLevelController):
    def __init__(self, vehicle, world_map, desired_gap_m=9.0, time_gap_s=0.6):
        super().__init__(vehicle)
        self.map = world_map
        self.desired_gap_m = float(desired_gap_m)
        self.time_gap_s = float(time_gap_s)
        self.path_cursor = 0
        self.lookahead_points = 2
        self.gap_gain = 0.55
        self.speed_damping = 0.20

    def _gap_target_speed(self, predecessor, ego_speed_mps, gap_distance_m):
        predecessor_vel = predecessor.get_velocity()
        predecessor_speed_mps = float(np.sqrt(predecessor_vel.x ** 2 + predecessor_vel.y ** 2 + predecessor_vel.z ** 2))
        desired_gap = self.desired_gap_m + self.time_gap_s * ego_speed_mps
        gap_error = gap_distance_m - desired_gap
        return max(0.0, predecessor_speed_mps + self.gap_gain * gap_error - self.speed_damping * max(0.0, ego_speed_mps - predecessor_speed_mps))

    def _select_path_waypoint(self, path_history):
        if not path_history:
            return self.map.get_waypoint(self.vehicle.get_location())

        location = self.vehicle.get_location()
        while self.path_cursor < len(path_history) - 1:
            waypoint = path_history[self.path_cursor]
            if location.distance(waypoint.transform.location) < 4.0:
                self.path_cursor += 1
            else:
                break

        target_idx = min(len(path_history) - 1, self.path_cursor + self.lookahead_points)
        return path_history[target_idx]

    def step(self, predecessor, path_history, gap_distance_m, override_waypoint=None):
        ego_vel = self.vehicle.get_velocity()
        ego_speed_mps = float(np.sqrt(ego_vel.x ** 2 + ego_vel.y ** 2 + ego_vel.z ** 2))
        target_waypoint = override_waypoint or self._select_path_waypoint(path_history)
        target_speed_mps = self._gap_target_speed(predecessor, ego_speed_mps, gap_distance_m)
        return self.run_step(target_speed_mps * 3.6, target_waypoint)

    def reset_path_cursor(self):
        self.path_cursor = 0


class DeterministicScenarioRunner(Node):
    def __init__(self):
        super().__init__('deterministic_scenario_runner')

        self.platoon_state = PlatoonState([0, 1, 2])
        self.truck_order = self.platoon_state.order
        self.maneuver_state = ManeuverState.IDLE
        self.reorder_direction = 'right'
        self.exiting_leader_id = -1
        self.successor_id = -1
        self.follower_id = -1
        self.promote_target_id = -1
        self.promote_original_leader_id = -1
        self.current_scenario_name = 'idle'
        self.current_scenario_direction = 'none'
        self.maneuver_start_time = None
        self.last_maneuver_duration_sec = 0.0

        self.target_velocity = float(os.getenv('DET_TARGET_VELOCITY_MPS', '16.0'))
        self.control_dt = float(os.getenv('DET_CONTROL_DT_SEC', '0.05'))
        self.auto_scenario_enabled = os.getenv('DET_AUTO_SCENARIO_ENABLED', '1').lower() in ('1', 'true', 'yes', 'on')
        self.auto_trigger_center_x = float(os.getenv('DET_AUTO_TRIGGER_CENTER_X', '-9.2'))
        self.auto_trigger_center_y = float(os.getenv('DET_AUTO_TRIGGER_CENTER_Y', '-93.3'))
        self.auto_trigger_half_x_m = float(os.getenv('DET_AUTO_TRIGGER_HALF_X_M', '3.0'))
        self.auto_trigger_half_y_m = float(os.getenv('DET_AUTO_TRIGGER_HALF_Y_M', '12.0'))
        self.auto_next_direction = os.getenv('DET_AUTO_NEXT_DIRECTION', 'right')
        self.reorder_safe_gap_distance_m = float(os.getenv('DET_REORDER_SAFE_GAP_DISTANCE_M', '25.0'))
        self.promote_safe_reentry_distance_m = float(os.getenv('DET_PROMOTE_SAFE_REENTRY_DISTANCE_M', '10.0'))
        self.rejoin_distance_min_m = float(os.getenv('DET_REJOIN_DISTANCE_MIN_M', '10.0'))
        self.rejoin_distance_max_m = float(os.getenv('DET_REJOIN_DISTANCE_MAX_M', '18.0'))
        self.max_lateral_offset_m = float(os.getenv('DET_MAX_LATERAL_OFFSET_M', '1.0'))
        self.change_timeout_sec = float(os.getenv('DET_LANE_CHANGE_TIMEOUT_SEC', '8.0'))
        self.cooldown_sec = float(os.getenv('DET_SCENARIO_COOLDOWN_SEC', '1.0'))
        self.experiment_tag = os.getenv('DET_EXPERIMENT_TAG', 'deterministic_default')
        self.scenario_log_dir = os.getenv('DET_SCENARIO_LOG_DIR', '/home/tmo/ros2_ws/log/scenario')

        self.client = None
        self.world = None
        self.world_map = None
        self.original_settings = None
        self.carla_actors = {}
        self.controllers = {}
        self.current_velocities = {i: 0.0 for i in range(3)}
        self.lead_waypoint_history = deque(maxlen=600)
        self.agent = ScriptedZoneTriggerAgent(initial_direction=self.auto_next_direction) if self.auto_scenario_enabled else PassiveScenarioAgent()
        self._scenario_log_file = None
        self._scenario_log_writer = None
        self._rejoin_timer = None

        self.order_publisher = self.create_publisher(Int32MultiArray, '/platoon_order', 10)
        self.maneuver_elapsed_publisher = self.create_publisher(Float32, '/platoon_maneuver_elapsed_sec', 10)
        self.maneuver_last_duration_publisher = self.create_publisher(Float32, '/platoon_maneuver_last_duration_sec', 10)
        self.start_reorder_service = self.create_service(SetBool, 'det_start_reorder', self.start_reorder_callback)
        self.start_promote_second_service = self.create_service(SetBool, 'det_start_promote_second', self.start_promote_second_callback)
        self.start_promote_third_service = self.create_service(SetBool, 'det_start_promote_third', self.start_promote_third_callback)

        self._connect_to_carla()
        self._setup_scenario_logger()
        self._publish_truck_order()

        self.metrics_timer = self.create_timer(0.1, self._publish_maneuver_metrics)
        self.control_timer = self.create_timer(self.control_dt, self.control_loop)

    def _connect_to_carla(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.world_map = self.world.get_map()
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.control_dt
        settings.substepping = True
        settings.max_substep_delta_time = min(0.01, self.control_dt)
        settings.max_substeps = max(1, int(np.ceil(self.control_dt / max(1e-3, settings.max_substep_delta_time))))
        self.world.apply_settings(settings)
        self._discover_actors()
        self._setup_controllers()
        self.world.tick()

    def _discover_actors(self):
        self.carla_actors = {}
        for actor in self.world.get_actors().filter('vehicle.*'):
            role_name = actor.attributes.get('role_name', '')
            if not role_name.startswith('truck'):
                continue
            try:
                truck_id = int(role_name.replace('truck', ''))
            except ValueError:
                continue
            if truck_id in range(3):
                actor.set_autopilot(False)
                self.carla_actors[truck_id] = actor
                self.get_logger().info(f"Found CARLA actor '{role_name}' for deterministic runner.")

    def _setup_controllers(self):
        self.controllers = {}
        leader_id = self.platoon_state.leader_id
        if leader_id is None or len(self.carla_actors) != 3:
            return
        self.controllers[leader_id] = LeadNavigator(self.carla_actors[leader_id], self.world_map, self.target_velocity)
        for truck_id in self.platoon_state.order[1:]:
            self.controllers[truck_id] = FollowerPathController(self.carla_actors[truck_id], self.world_map)
        self._append_lead_waypoint(force=True)

    def _setup_scenario_logger(self):
        os.makedirs(self.scenario_log_dir, exist_ok=True)
        path = os.path.join(self.scenario_log_dir, 'deterministic_scenario_runs.csv')
        file_exists = os.path.exists(path) and os.path.getsize(path) > 0
        self._scenario_log_file = open(path, 'a', newline='')
        self._scenario_log_writer = csv.writer(self._scenario_log_file)
        if not file_exists:
            self._scenario_log_writer.writerow([
                'timestamp',
                'experiment_tag',
                'scenario',
                'direction',
                'duration_sec',
                'final_order',
                'target_velocity_mps',
                'control_dt_sec',
            ])
            self._scenario_log_file.flush()

    def _write_scenario_log_row(self):
        if self._scenario_log_writer is None:
            return
        self._scenario_log_writer.writerow([
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            self.experiment_tag,
            self.current_scenario_name,
            self.current_scenario_direction,
            f'{self.last_maneuver_duration_sec:.3f}',
            '-'.join(str(x) for x in self.truck_order),
            f'{self.target_velocity:.3f}',
            f'{self.control_dt:.3f}',
        ])
        self._scenario_log_file.flush()

    def _publish_truck_order(self):
        self.truck_order = self.platoon_state.order
        msg = Int32MultiArray()
        msg.data = list(self.truck_order)
        self.order_publisher.publish(msg)

    def _publish_maneuver_metrics(self):
        elapsed = 0.0
        if self.maneuver_start_time is not None and self.maneuver_state != ManeuverState.IDLE:
            elapsed = max(0.0, time.monotonic() - self.maneuver_start_time)
        msg = Float32()
        msg.data = float(elapsed)
        self.maneuver_elapsed_publisher.publish(msg)
        last = Float32()
        last.data = float(self.last_maneuver_duration_sec)
        self.maneuver_last_duration_publisher.publish(last)

    def _append_lead_waypoint(self, force=False):
        leader_id = self.platoon_state.leader_id
        if leader_id is None:
            return
        waypoint = self.get_vehicle_waypoint(leader_id)
        if waypoint is None:
            return
        if not force and self.lead_waypoint_history:
            last = self.lead_waypoint_history[-1]
            if waypoint.transform.location.distance(last.transform.location) < 1.5:
                return
        self.lead_waypoint_history.append(waypoint)

    def get_vehicle_transform(self, truck_id):
        actor = self.carla_actors.get(truck_id)
        return actor.get_transform() if actor is not None else None

    def get_vehicle_waypoint(self, truck_id):
        actor = self.carla_actors.get(truck_id)
        if actor is None:
            return None
        return self.world_map.get_waypoint(actor.get_location())

    def _lane_ref(self, waypoint):
        if waypoint is None:
            return None
        return (waypoint.road_id, waypoint.section_id, waypoint.lane_id)

    def _find_adjacent_driving_lane(self, waypoint, direction):
        if waypoint is None:
            return None
        candidate = waypoint.get_right_lane() if direction == 'right' else waypoint.get_left_lane()
        visited = set()
        while candidate is not None:
            lane_ref = self._lane_ref(candidate)
            if lane_ref in visited:
                break
            visited.add(lane_ref)
            if candidate.lane_type == carla.LaneType.Driving:
                return candidate
            candidate = candidate.get_right_lane() if direction == 'right' else candidate.get_left_lane()
        return None

    def _distance_to_previous_truck(self, follower_id):
        order = self.platoon_state.order
        if follower_id not in order:
            return None
        rank = order.index(follower_id)
        if rank <= 0:
            return None
        leader_id = order[rank - 1]
        follower_tf = self.get_vehicle_transform(follower_id)
        leader_tf = self.get_vehicle_transform(leader_id)
        follower_actor = self.carla_actors.get(follower_id)
        leader_actor = self.carla_actors.get(leader_id)
        if None in (follower_tf, leader_tf, follower_actor, leader_actor):
            return None
        relative_vec = leader_tf.location - follower_tf.location
        forward_distance = float(relative_vec.dot(follower_tf.get_forward_vector()))
        if forward_distance <= 0.0:
            return None
        follower_extent = float(getattr(follower_actor.bounding_box.extent, 'x', 0.0))
        leader_extent = float(getattr(leader_actor.bounding_box.extent, 'x', 0.0))
        return max(0.0, forward_distance - follower_extent - leader_extent)

    def _is_in_auto_trigger_zone(self, location):
        return (
            abs(float(location.x) - self.auto_trigger_center_x) <= self.auto_trigger_half_x_m
            and abs(float(location.y) - self.auto_trigger_center_y) <= self.auto_trigger_half_y_m
        )

    def _maybe_trigger_auto_scenario(self):
        if not self.auto_scenario_enabled or self.maneuver_state != ManeuverState.IDLE:
            return
        leader_id = self.platoon_state.leader_id
        leader_tf = self.get_vehicle_transform(leader_id) if leader_id is not None else None
        if leader_tf is None:
            return
        command = self.agent.decide(
            PlatoonSnapshot(
                platoon_id='main',
                order=self.platoon_state.order,
                leader_speed_mps=self.current_velocities.get(leader_id, 0.0),
                in_trigger_zone=self._is_in_auto_trigger_zone(leader_tf.location),
            ),
            WorldSnapshot(
                time_s=time.monotonic(),
                phase='IDLE',
            ),
        )
        if command.kind == ScenarioCommandType.START_REORDER:
            self.start_reorder_maneuver(command.direction)

    def _scenario_target_speed(self, truck_id, rank):
        if self.maneuver_state in (ManeuverState.IDLE, ManeuverState.COOLDOWN, ManeuverState.REORDER_COMPLETE):
            return self.target_velocity
        if self.maneuver_state in (ManeuverState.PROMOTE_TARGET_EXITS, ManeuverState.PROMOTE_PLATOON_CREATES_GAP, ManeuverState.PROMOTE_TARGET_REENTERS):
            if truck_id == self.promote_target_id:
                return self.target_velocity * (1.2 if rank == 1 else 1.1)
            if self.maneuver_state in (ManeuverState.PROMOTE_PLATOON_CREATES_GAP, ManeuverState.PROMOTE_TARGET_REENTERS):
                return self.target_velocity * 0.8
        if truck_id == self.exiting_leader_id and self.maneuver_state in (
            ManeuverState.LEADER_CREATES_GAP,
            ManeuverState.SUCCESSOR_ENTERS_GAP,
            ManeuverState.FOLLOWER_ENTERS_GAP,
        ):
            return self.target_velocity * 0.65
        return self.target_velocity

    def _resolve_lane_change_waypoint(self, truck_id):
        truck = self.platoon_state.truck(truck_id)
        if truck.target_lane == 'center':
            return None
        current_wp = self.get_vehicle_waypoint(truck_id)
        adjacent_wp = self._find_adjacent_driving_lane(current_wp, truck.target_lane)
        if adjacent_wp is None:
            return None
        candidates = adjacent_wp.next(8.0)
        return candidates[0] if candidates else adjacent_wp

    def _is_lane_change_complete(self, truck_id):
        truck = self.platoon_state.truck(truck_id)
        if truck.target_lane == 'center':
            return True
        current_wp = self.get_vehicle_waypoint(truck_id)
        if current_wp is not None and truck.target_lane_ref is not None and self._lane_ref(current_wp) == truck.target_lane_ref:
            return True
        if truck.lane_change_request_ts is not None and (time.monotonic() - truck.lane_change_request_ts) > self.change_timeout_sec:
            self.get_logger().warn(f'Truck {truck_id} lane change timeout.')
            return True
        return False

    def _start_lane_change(self, truck_id, direction):
        current_wp = self.get_vehicle_waypoint(truck_id)
        adjacent_wp = self._find_adjacent_driving_lane(current_wp, direction)
        if adjacent_wp is None:
            self.get_logger().warn(f'Truck {truck_id}: no lane available to the {direction}.')
            return False
        self.platoon_state.start_lane_change(truck_id, direction, self._lane_ref(adjacent_wp), time.monotonic())
        return True

    def _reset_lane_change(self, truck_id):
        self.platoon_state.reset_lane_change(truck_id)

    def _start_maneuver_timer(self):
        self.maneuver_start_time = time.monotonic()

    def start_reorder_callback(self, request, response):
        success = self.start_reorder_maneuver('left' if request.data else 'right')
        response.success = success
        response.message = 'ok' if success else 'rejected'
        return response

    def start_promote_second_callback(self, request, response):
        target_id = self.platoon_state.order[1] if len(self.platoon_state.order) > 1 else -1
        success = self.start_promote_maneuver(target_id, 'left' if request.data else 'right')
        response.success = success
        response.message = 'ok' if success else 'rejected'
        return response

    def start_promote_third_callback(self, request, response):
        target_id = self.platoon_state.order[2] if len(self.platoon_state.order) > 2 else -1
        success = self.start_promote_maneuver(target_id, 'left' if request.data else 'right')
        response.success = success
        response.message = 'ok' if success else 'rejected'
        return response

    def start_reorder_maneuver(self, direction):
        order = self.platoon_state.order
        if self.maneuver_state != ManeuverState.IDLE or len(order) < 3:
            return False
        self.current_scenario_name = 'reorder'
        self.current_scenario_direction = direction
        self.reorder_direction = direction
        self.exiting_leader_id, self.successor_id, self.follower_id = order[0], order[1], order[2]
        if not self._start_lane_change(self.exiting_leader_id, direction):
            return False
        self.maneuver_state = ManeuverState.LEADER_EXITS_LANE
        self._start_maneuver_timer()
        return True

    def start_promote_maneuver(self, target_id, direction):
        order = self.platoon_state.order
        if self.maneuver_state != ManeuverState.IDLE or target_id not in order[1:]:
            return False
        self.current_scenario_name = f'promote_truck{target_id}'
        self.current_scenario_direction = direction
        self.reorder_direction = direction
        self.promote_target_id = target_id
        self.promote_original_leader_id = order[0]
        if not self._start_lane_change(target_id, direction):
            return False
        self.maneuver_state = ManeuverState.PROMOTE_TARGET_EXITS
        self._start_maneuver_timer()
        return True

    def _start_rejoin_check(self):
        if self._rejoin_timer is not None:
            try:
                self._rejoin_timer.cancel()
            except Exception:
                pass
        self._rejoin_tick()

    def _rejoin_tick(self):
        if self.maneuver_state != ManeuverState.LEADER_REENTERS_LANE:
            self._rejoin_timer = None
            return
        leader_tf = self.get_vehicle_transform(self.exiting_leader_id)
        follower_tf = self.get_vehicle_transform(self.follower_id)
        if leader_tf is not None and follower_tf is not None:
            relative_vec = leader_tf.location - follower_tf.location
            forward_dist = float(relative_vec.dot(follower_tf.get_forward_vector()))
            lateral_dist = abs(float(relative_vec.dot(follower_tf.get_right_vector())))
            if (
                forward_dist < 0.0
                and self.rejoin_distance_min_m <= abs(forward_dist) <= self.rejoin_distance_max_m
                and lateral_dist <= self.max_lateral_offset_m
            ):
                reenter_dir = 'left' if self.reorder_direction == 'right' else 'right'
                self._start_lane_change(self.exiting_leader_id, reenter_dir)
                self._rejoin_timer = None
                return
        self._rejoin_timer = threading.Timer(0.1, self._rejoin_tick)
        self._rejoin_timer.start()

    def manage_maneuver(self):
        if self.maneuver_state == ManeuverState.IDLE:
            return
        if self.maneuver_state == ManeuverState.LEADER_EXITS_LANE:
            if self._is_lane_change_complete(self.exiting_leader_id):
                self._reset_lane_change(self.exiting_leader_id)
                self.maneuver_state = ManeuverState.LEADER_CREATES_GAP
        elif self.maneuver_state == ManeuverState.LEADER_CREATES_GAP:
            leader_tf = self.get_vehicle_transform(self.exiting_leader_id)
            successor_tf = self.get_vehicle_transform(self.successor_id)
            if leader_tf is None or successor_tf is None:
                return
            relative_vec = leader_tf.location - successor_tf.location
            forward_dist = float(relative_vec.dot(successor_tf.get_forward_vector()))
            if forward_dist < 0.0 and abs(forward_dist) >= self.reorder_safe_gap_distance_m:
                if self._start_lane_change(self.successor_id, self.reorder_direction):
                    self.maneuver_state = ManeuverState.SUCCESSOR_ENTERS_GAP
        elif self.maneuver_state == ManeuverState.SUCCESSOR_ENTERS_GAP:
            if self._is_lane_change_complete(self.successor_id):
                self._reset_lane_change(self.successor_id)
                if self._start_lane_change(self.follower_id, self.reorder_direction):
                    self.maneuver_state = ManeuverState.FOLLOWER_ENTERS_GAP
        elif self.maneuver_state == ManeuverState.FOLLOWER_ENTERS_GAP:
            if self._is_lane_change_complete(self.follower_id):
                self._reset_lane_change(self.follower_id)
                self.maneuver_state = ManeuverState.LEADER_REENTERS_LANE
                self._start_rejoin_check()
        elif self.maneuver_state == ManeuverState.LEADER_REENTERS_LANE:
            if self._is_lane_change_complete(self.exiting_leader_id):
                self._reset_lane_change(self.exiting_leader_id)
                self.finalize_maneuver(reorder=True)
        elif self.maneuver_state == ManeuverState.PROMOTE_TARGET_EXITS:
            if self._is_lane_change_complete(self.promote_target_id):
                self._reset_lane_change(self.promote_target_id)
                self.maneuver_state = ManeuverState.PROMOTE_PLATOON_CREATES_GAP
        elif self.maneuver_state == ManeuverState.PROMOTE_PLATOON_CREATES_GAP:
            target_tf = self.get_vehicle_transform(self.promote_target_id)
            leader_tf = self.get_vehicle_transform(self.promote_original_leader_id)
            if target_tf is None or leader_tf is None:
                return
            relative_vec = target_tf.location - leader_tf.location
            forward_distance = float(relative_vec.dot(leader_tf.get_forward_vector()))
            if forward_distance >= self.promote_safe_reentry_distance_m:
                reenter_dir = 'left' if self.reorder_direction == 'right' else 'right'
                if self._start_lane_change(self.promote_target_id, reenter_dir):
                    self.maneuver_state = ManeuverState.PROMOTE_TARGET_REENTERS
        elif self.maneuver_state == ManeuverState.PROMOTE_TARGET_REENTERS:
            if self._is_lane_change_complete(self.promote_target_id):
                self._reset_lane_change(self.promote_target_id)
                self.finalize_maneuver(reorder=False)

    def finalize_maneuver(self, reorder: bool):
        if self._rejoin_timer is not None:
            try:
                self._rejoin_timer.cancel()
            except Exception:
                pass
            self._rejoin_timer = None
        if reorder:
            self.platoon_state.rotate_leader_to_tail()
        elif self.promote_target_id != -1:
            self.platoon_state.promote_to_lead(self.promote_target_id)
        self._publish_truck_order()
        if self.maneuver_start_time is not None:
            self.last_maneuver_duration_sec = max(0.0, time.monotonic() - self.maneuver_start_time)
            self.maneuver_start_time = None
            self._write_scenario_log_row()
        self.platoon_state.reset_all_lane_changes()
        for controller in self.controllers.values():
            if isinstance(controller, FollowerPathController):
                controller.reset_path_cursor()
        self.maneuver_state = ManeuverState.COOLDOWN
        threading.Timer(self.cooldown_sec, self.reset_maneuver_variables).start()

    def reset_maneuver_variables(self):
        self.maneuver_state = ManeuverState.IDLE
        self.exiting_leader_id = -1
        self.successor_id = -1
        self.follower_id = -1
        self.promote_target_id = -1
        self.promote_original_leader_id = -1
        self.current_scenario_name = 'idle'
        self.current_scenario_direction = 'none'
        self._publish_maneuver_metrics()

    def _update_velocity_cache(self):
        for truck_id, actor in self.carla_actors.items():
            vel = actor.get_velocity()
            self.current_velocities[truck_id] = float(np.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2))

    def control_loop(self):
        if self.world is None or len(self.carla_actors) != 3:
            return
        self._update_velocity_cache()
        self._append_lead_waypoint()
        self._maybe_trigger_auto_scenario()
        self.manage_maneuver()

        order = self.platoon_state.order
        for rank, truck_id in enumerate(order):
            actor = self.carla_actors[truck_id]
            lane_change_target = self._resolve_lane_change_waypoint(truck_id)
            if rank == 0:
                controller = self.controllers[truck_id]
                control = controller.step(
                    override_waypoint=lane_change_target,
                    target_speed_mps=self._scenario_target_speed(truck_id, rank),
                )
            else:
                predecessor = self.carla_actors[order[rank - 1]]
                gap_distance = self._distance_to_previous_truck(truck_id)
                if gap_distance is None:
                    gap_distance = 30.0
                controller = self.controllers[truck_id]
                control = controller.step(
                    predecessor=predecessor,
                    path_history=self.lead_waypoint_history,
                    gap_distance_m=gap_distance,
                    override_waypoint=lane_change_target,
                )
            actor.apply_control(control)

        self.world.tick()

    def destroy_node(self):
        try:
            if self.world is not None and self.original_settings is not None:
                self.world.apply_settings(self.original_settings)
        except Exception:
            pass
        if self._scenario_log_file is not None:
            try:
                self._scenario_log_file.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DeterministicScenarioRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
