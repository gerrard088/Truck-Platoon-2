import csv
import os
import threading
import time

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


class TmScenarioRunner(Node):
    def __init__(self):
        super().__init__('tm_scenario_runner')

        self.truck_order = [0, 1, 2]
        self.platoon_state = PlatoonState(self.truck_order)
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

        self.target_velocity = float(os.getenv('TM_SCENARIO_TARGET_VELOCITY_MPS', '19.5'))
        self.autopilot_follow_distance_m = float(os.getenv('TM_AUTOPILOT_FOLLOW_DISTANCE_M', '8.0'))
        self.tm_port = int(os.getenv('TM_TRAFFIC_MANAGER_PORT', '8000'))
        self.tm_seed = int(os.getenv('TM_TRAFFIC_MANAGER_SEED', '42'))
        self.ignore_lights_percent = float(os.getenv('TM_IGNORE_LIGHTS_PERCENT', '100.0'))
        self.ignore_signs_percent = float(os.getenv('TM_IGNORE_SIGNS_PERCENT', '100.0'))
        self.change_timeout_sec = float(os.getenv('TM_LANE_CHANGE_TIMEOUT_SEC', '8.0'))
        self.cooldown_sec = float(os.getenv('TM_SCENARIO_COOLDOWN_SEC', '1.0'))

        self.auto_scenario_enabled = os.getenv('TM_AUTO_SCENARIO_ENABLED', '1').lower() in ('1', 'true', 'yes', 'on')
        self.auto_trigger_center_x = float(os.getenv('TM_AUTO_TRIGGER_CENTER_X', '-9.2'))
        self.auto_trigger_center_y = float(os.getenv('TM_AUTO_TRIGGER_CENTER_Y', '-93.3'))
        self.auto_trigger_half_x_m = float(os.getenv('TM_AUTO_TRIGGER_HALF_X_M', '3.0'))
        self.auto_trigger_half_y_m = float(os.getenv('TM_AUTO_TRIGGER_HALF_Y_M', '12.0'))
        self.auto_trigger_latched = False
        self.auto_next_direction = os.getenv('TM_AUTO_NEXT_DIRECTION', 'right')

        self.reorder_safe_gap_distance_m = float(os.getenv('TM_REORDER_SAFE_GAP_DISTANCE_M', '25.0'))
        self.promote_safe_reentry_distance_m = float(os.getenv('TM_PROMOTE_SAFE_REENTRY_DISTANCE_M', '10.0'))
        self.rejoin_distance_min_m = float(os.getenv('TM_REJOIN_DISTANCE_MIN_M', '10.0'))
        self.rejoin_distance_max_m = float(os.getenv('TM_REJOIN_DISTANCE_MAX_M', '18.0'))
        self.max_lateral_offset_m = float(os.getenv('TM_MAX_LATERAL_OFFSET_M', '1.0'))

        self.scenario_log_dir = os.getenv('TM_SCENARIO_LOG_DIR', '/home/tmo/ros2_ws/log/scenario')
        self.experiment_tag = os.getenv('TM_EXPERIMENT_TAG', 'tm_default')
        self._scenario_log_file = None
        self._scenario_log_writer = None

        self.order_publisher = self.create_publisher(Int32MultiArray, '/platoon_order', 10)
        self.maneuver_elapsed_publisher = self.create_publisher(Float32, '/platoon_maneuver_elapsed_sec', 10)
        self.maneuver_last_duration_publisher = self.create_publisher(Float32, '/platoon_maneuver_last_duration_sec', 10)

        self.start_reorder_service = self.create_service(SetBool, 'tm_start_reorder', self.start_reorder_callback)
        self.start_promote_second_service = self.create_service(SetBool, 'tm_start_promote_second', self.start_promote_second_callback)
        self.start_promote_third_service = self.create_service(SetBool, 'tm_start_promote_third', self.start_promote_third_callback)

        self.world = None
        self.world_map = None
        self.client = None
        self.traffic_manager = None
        self.carla_actors = {}
        self.current_velocities = {i: 0.0 for i in range(3)}
        self.agent = ScriptedZoneTriggerAgent(initial_direction=self.auto_next_direction) if self.auto_scenario_enabled else PassiveScenarioAgent()

        self._rejoin_timer = None

        self._connect_to_carla()
        self._setup_scenario_logger()
        self._publish_truck_order()

        self.metrics_timer = self.create_timer(0.1, self._publish_maneuver_metrics)
        self.auto_trigger_timer = self.create_timer(0.1, self._maybe_trigger_auto_scenario)
        self.maneuver_timer = self.create_timer(0.2, self.manage_maneuver)
        self.autopilot_timer = self.create_timer(0.2, self._apply_autopilot_profiles)

    def _connect_to_carla(self):
        try:
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(5.0)
            self.world = self.client.get_world()
            self.world_map = self.world.get_map()
            self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
            self.traffic_manager.set_random_device_seed(self.tm_seed)
            self._discover_actors()
            self._setup_autopilot()
        except Exception as e:
            self.get_logger().error(f'Failed to initialize CARLA/TM: {e}')

    def _discover_actors(self):
        self.carla_actors = {}
        if self.world is None:
            return
        for actor in self.world.get_actors().filter('vehicle.*'):
            role_name = actor.attributes.get('role_name', '')
            if not role_name.startswith('truck'):
                continue
            try:
                truck_id = int(role_name.replace('truck', ''))
            except ValueError:
                continue
            if truck_id in range(3):
                self.carla_actors[truck_id] = actor
                self.get_logger().info(f"Found CARLA actor '{role_name}'.")
        if len(self.carla_actors) != 3:
            self.get_logger().warn(f'Found {len(self.carla_actors)} truck actors, expected 3.')

    def _setup_autopilot(self):
        if self.traffic_manager is None:
            return
        for truck_id, actor in self.carla_actors.items():
            try:
                actor.set_autopilot(True, self.tm_port)
                self.traffic_manager.auto_lane_change(actor, False)
                self.traffic_manager.distance_to_leading_vehicle(actor, self.autopilot_follow_distance_m)
                self.traffic_manager.ignore_lights_percentage(actor, self.ignore_lights_percent)
                self.traffic_manager.ignore_signs_percentage(actor, self.ignore_signs_percent)
                self._apply_speed_target(truck_id, self.target_velocity)
                self.get_logger().info(f'Autopilot enabled for truck{truck_id}.')
            except Exception as e:
                self.get_logger().error(f'Failed to configure autopilot for truck{truck_id}: {e}')

    def _setup_scenario_logger(self):
        try:
            os.makedirs(self.scenario_log_dir, exist_ok=True)
            path = os.path.join(self.scenario_log_dir, 'tm_scenario_runs.csv')
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
                    'tm_seed',
                    'tm_follow_distance_m',
                    'ignore_lights_percent',
                    'ignore_signs_percent',
                ])
                self._scenario_log_file.flush()
        except Exception as e:
            self.get_logger().error(f'Failed to initialize scenario logger: {e}')

    def _write_scenario_log_row(self):
        if self._scenario_log_writer is None:
            return
        try:
            self._scenario_log_writer.writerow([
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                self.experiment_tag,
                self.current_scenario_name,
                self.current_scenario_direction,
                f'{self.last_maneuver_duration_sec:.3f}',
                '-'.join(str(x) for x in self.truck_order),
                f'{self.target_velocity:.3f}',
                self.tm_seed,
                f'{self.autopilot_follow_distance_m:.3f}',
                f'{self.ignore_lights_percent:.1f}',
                f'{self.ignore_signs_percent:.1f}',
            ])
            self._scenario_log_file.flush()
        except Exception as e:
            self.get_logger().error(f'Failed to write scenario log row: {e}')

    def _publish_truck_order(self):
        msg = Int32MultiArray()
        self.truck_order = self.platoon_state.order
        msg.data = list(self.truck_order)
        self.order_publisher.publish(msg)

    def _publish_maneuver_metrics(self):
        elapsed = 0.0
        if self.maneuver_start_time is not None and self.maneuver_state != ManeuverState.IDLE:
            elapsed = max(0.0, time.monotonic() - self.maneuver_start_time)
        elapsed_msg = Float32()
        elapsed_msg.data = float(elapsed)
        self.maneuver_elapsed_publisher.publish(elapsed_msg)
        duration_msg = Float32()
        duration_msg.data = float(self.last_maneuver_duration_sec)
        self.maneuver_last_duration_publisher.publish(duration_msg)

    def _velocity_to_tm_speed_difference(self, actor, target_velocity_mps):
        target_kmh = max(1.0, float(target_velocity_mps) * 3.6)
        try: ros2 launch tm_experiment_control tm_experiment.launch.py map:=Town04_Opt
  experiment_tag:=tm_refactored


            speed_limit_kmh = float(actor.get_speed_limit())
        except Exception:
            speed_limit_kmh = target_kmh
        if speed_limit_kmh <= 1.0:
            speed_limit_kmh = target_kmh
        speed_diff = ((speed_limit_kmh - target_kmh) / speed_limit_kmh) * 100.0
        return float(np.clip(speed_diff, -50.0, 80.0))

    def _apply_speed_target(self, truck_id, target_velocity_mps):
        actor = self.carla_actors.get(truck_id)
        if actor is None or self.traffic_manager is None:
            return
        try:
            self.traffic_manager.vehicle_percentage_speed_difference(
                actor,
                self._velocity_to_tm_speed_difference(actor, target_velocity_mps),
            )
        except Exception as e:
            self.get_logger().warn(f'Failed to set speed target for truck{truck_id}: {e}')

    def _apply_autopilot_profiles(self):
        if self.traffic_manager is None:
            return
        self.truck_order = self.platoon_state.order
        for index, truck_id in enumerate(self.truck_order):
            actor = self.carla_actors.get(truck_id)
            if actor is None:
                continue
            target_velocity = self._scenario_target_velocity(truck_id, index)
            self._apply_speed_target(truck_id, target_velocity)
            try:
                self.traffic_manager.auto_lane_change(actor, False)
                self.traffic_manager.distance_to_leading_vehicle(actor, self.autopilot_follow_distance_m)
            except Exception as e:
                self.get_logger().warn(f'Failed to refresh TM profile for truck{truck_id}: {e}')
            self._update_velocity_cache(truck_id)

    def _update_velocity_cache(self, truck_id):
        actor = self.carla_actors.get(truck_id)
        if actor is None:
            return
        try:
            vel = actor.get_velocity()
            self.current_velocities[truck_id] = float(np.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z))
        except Exception:
            pass

    def _scenario_target_velocity(self, truck_id, rank):
        if self.maneuver_state in (ManeuverState.IDLE, ManeuverState.COOLDOWN, ManeuverState.REORDER_COMPLETE):
            return self.target_velocity
        if self.maneuver_state in (ManeuverState.PROMOTE_TARGET_EXITS, ManeuverState.PROMOTE_PLATOON_CREATES_GAP, ManeuverState.PROMOTE_TARGET_REENTERS):
            if truck_id == self.promote_target_id:
                return self.target_velocity * (1.2 if rank == 1 else 1.1)
            if self.maneuver_state in (ManeuverState.PROMOTE_PLATOON_CREATES_GAP, ManeuverState.PROMOTE_TARGET_REENTERS):
                return self.target_velocity * 0.8
            return self.target_velocity
        if truck_id == self.exiting_leader_id and self.maneuver_state in (
            ManeuverState.LEADER_CREATES_GAP,
            ManeuverState.SUCCESSOR_ENTERS_GAP,
            ManeuverState.FOLLOWER_ENTERS_GAP,
        ):
            return self.target_velocity * 0.65
        if truck_id in (self.successor_id, self.follower_id) and self.maneuver_state in (
            ManeuverState.SUCCESSOR_ENTERS_GAP,
            ManeuverState.FOLLOWER_ENTERS_GAP,
        ):
            return self.target_velocity * 1.05
        return self.target_velocity

    def get_vehicle_transform(self, truck_id):
        actor = self.carla_actors.get(truck_id)
        if actor is None:
            return None
        try:
            return actor.get_transform()
        except Exception:
            return None

    def get_vehicle_waypoint(self, truck_id):
        actor = self.carla_actors.get(truck_id)
        if actor is None or self.world_map is None:
            return None
        try:
            return self.world_map.get_waypoint(actor.get_location())
        except Exception:
            return None

    def _lane_ref(self, waypoint):
        if waypoint is None:
            return None
        return (waypoint.road_id, waypoint.section_id, waypoint.lane_id)

    def _distance_between_locations(self, loc_a, loc_b):
        dx = float(loc_a.x - loc_b.x)
        dy = float(loc_a.y - loc_b.y)
        dz = float(loc_a.z - loc_b.z)
        return float(np.sqrt(dx * dx + dy * dy + dz * dz))

    def _project_forward_distance(self, transform, target_location):
        vehicle_loc = transform.location
        dx = float(target_location.x - vehicle_loc.x)
        dy = float(target_location.y - vehicle_loc.y)
        dz = float(target_location.z - vehicle_loc.z)
        forward = transform.get_forward_vector()
        return float(dx * forward.x + dy * forward.y + dz * forward.z)

    def _distance_to_previous_truck(self, follower_id):
        self.truck_order = self.platoon_state.order
        if follower_id not in self.truck_order:
            return None
        rank = self.truck_order.index(follower_id)
        if rank <= 0:
            return None
        leader_id = self.truck_order[rank - 1]
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
        self.truck_order = self.platoon_state.order
        if not self.auto_scenario_enabled or self.maneuver_state != ManeuverState.IDLE or not self.truck_order:
            return
        leader_tf = self.get_vehicle_transform(self.truck_order[0])
        if leader_tf is None:
            return
        in_zone = self._is_in_auto_trigger_zone(leader_tf.location)
        snapshot = PlatoonSnapshot(
            platoon_id="main",
            order=self.truck_order,
            leader_speed_mps=self.current_velocities.get(self.truck_order[0], 0.0),
            in_trigger_zone=in_zone,
        )
        world_state = WorldSnapshot(
            time_s=time.monotonic(),
            phase="IDLE" if self.maneuver_state == ManeuverState.IDLE else "ACTIVE",
        )
        command = self.agent.decide(snapshot, world_state)
        if command.kind == ScenarioCommandType.START_REORDER:
            self.start_reorder_maneuver(command.direction)

    def _start_maneuver_timer(self):
        self.maneuver_start_time = time.monotonic()

    def _is_lane_change_complete(self, truck_id):
        truck = self.platoon_state.truck(truck_id)
        direction = truck.target_lane
        if direction == 'center':
            return True
        current_wp = self.get_vehicle_waypoint(truck_id)
        target_ref = truck.target_lane_ref
        if current_wp is not None and target_ref is not None and self._lane_ref(current_wp) == target_ref:
            return True
        request_ts = truck.lane_change_request_ts
        if request_ts is not None and (time.monotonic() - request_ts) > self.change_timeout_sec:
            self.get_logger().warn(f'Truck {truck_id} lane change timeout; accepting current state.')
            return True
        return False

    def _force_lane_change(self, truck_id, direction):
        actor = self.carla_actors.get(truck_id)
        current_wp = self.get_vehicle_waypoint(truck_id)
        if actor is None or current_wp is None or self.traffic_manager is None:
            return False
        adjacent_wp = current_wp.get_right_lane() if direction == 'right' else current_wp.get_left_lane()
        while adjacent_wp is not None and adjacent_wp.lane_type != carla.LaneType.Driving:
            adjacent_wp = adjacent_wp.get_right_lane() if direction == 'right' else adjacent_wp.get_left_lane()
        if adjacent_wp is None:
            self.get_logger().warn(f'Truck {truck_id} has no driving lane to the {direction}.')
            return False
        try:
            self.traffic_manager.auto_lane_change(actor, False)
            self.traffic_manager.force_lane_change(actor, direction == 'right')
            self.platoon_state.start_lane_change(
                truck_id,
                direction,
                self._lane_ref(adjacent_wp),
                time.monotonic(),
            )
            self.get_logger().info(f'Truck {truck_id} force lane change -> {direction}')
            return True
        except Exception as e:
            self.get_logger().error(f'Failed to force lane change for truck{truck_id}: {e}')
            return False

    def _reset_lane_state(self, truck_id):
        self.platoon_state.reset_lane_change(truck_id)

    def start_reorder_callback(self, request, response):
        direction = 'left' if request.data else 'right'
        success = self.start_reorder_maneuver(direction)
        response.success = success
        response.message = f'reorder {direction} started' if success else 'reorder rejected'
        return response

    def start_promote_second_callback(self, request, response):
        direction = 'left' if request.data else 'right'
        target_id = self.truck_order[1] if len(self.truck_order) > 1 else -1
        success = self.start_promote_maneuver(target_id, direction)
        response.success = success
        response.message = f'promote second {direction} started' if success else 'promote second rejected'
        return response

    def start_promote_third_callback(self, request, response):
        direction = 'left' if request.data else 'right'
        target_id = self.truck_order[2] if len(self.truck_order) > 2 else -1
        success = self.start_promote_maneuver(target_id, direction)
        response.success = success
        response.message = f'promote third {direction} started' if success else 'promote third rejected'
        return response

    def start_reorder_maneuver(self, direction):
        self.truck_order = self.platoon_state.order
        if self.maneuver_state != ManeuverState.IDLE or len(self.truck_order) < 3:
            return False
        self.current_scenario_name = 'reorder'
        self.current_scenario_direction = direction
        self.reorder_direction = direction
        self.exiting_leader_id = self.truck_order[0]
        self.successor_id = self.truck_order[1]
        self.follower_id = self.truck_order[2]
        if not self._force_lane_change(self.exiting_leader_id, direction):
            return False
        self.maneuver_state = ManeuverState.LEADER_EXITS_LANE
        self._start_maneuver_timer()
        return True

    def start_promote_maneuver(self, target_id, direction):
        self.truck_order = self.platoon_state.order
        if self.maneuver_state != ManeuverState.IDLE or target_id not in self.truck_order[1:]:
            return False
        self.current_scenario_name = f'promote_truck{target_id}'
        self.current_scenario_direction = direction
        self.reorder_direction = direction
        self.promote_target_id = target_id
        self.promote_original_leader_id = self.truck_order[0]
        if not self._force_lane_change(target_id, direction):
            return False
        self.maneuver_state = ManeuverState.PROMOTE_TARGET_EXITS
        self._start_maneuver_timer()
        return True

    def manage_maneuver(self):
        if self.maneuver_state == ManeuverState.IDLE:
            return

        if self.maneuver_state == ManeuverState.LEADER_EXITS_LANE:
            if self._is_lane_change_complete(self.exiting_leader_id):
                self._reset_lane_state(self.exiting_leader_id)
                self.maneuver_state = ManeuverState.LEADER_CREATES_GAP

        elif self.maneuver_state == ManeuverState.LEADER_CREATES_GAP:
            leader_tf = self.get_vehicle_transform(self.exiting_leader_id)
            successor_tf = self.get_vehicle_transform(self.successor_id)
            if leader_tf is None or successor_tf is None:
                return
            relative_vec = leader_tf.location - successor_tf.location
            forward_dist = float(relative_vec.dot(successor_tf.get_forward_vector()))
            if forward_dist < 0.0 and abs(forward_dist) >= self.reorder_safe_gap_distance_m:
                if self._force_lane_change(self.successor_id, self.reorder_direction):
                    self.maneuver_state = ManeuverState.SUCCESSOR_ENTERS_GAP

        elif self.maneuver_state == ManeuverState.SUCCESSOR_ENTERS_GAP:
            if self._is_lane_change_complete(self.successor_id):
                self._reset_lane_state(self.successor_id)
                if self._force_lane_change(self.follower_id, self.reorder_direction):
                    self.maneuver_state = ManeuverState.FOLLOWER_ENTERS_GAP

        elif self.maneuver_state == ManeuverState.FOLLOWER_ENTERS_GAP:
            if self._is_lane_change_complete(self.follower_id):
                self._reset_lane_state(self.follower_id)
                self.maneuver_state = ManeuverState.LEADER_REENTERS_LANE
                self._start_rejoin_check()

        elif self.maneuver_state == ManeuverState.LEADER_REENTERS_LANE:
            if self._is_lane_change_complete(self.exiting_leader_id):
                self._reset_lane_state(self.exiting_leader_id)
                self.maneuver_state = ManeuverState.REORDER_COMPLETE
                self.finalize_maneuver()

        elif self.maneuver_state == ManeuverState.PROMOTE_TARGET_EXITS:
            if self._is_lane_change_complete(self.promote_target_id):
                self._reset_lane_state(self.promote_target_id)
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
                if self._force_lane_change(self.promote_target_id, reenter_dir):
                    self.maneuver_state = ManeuverState.PROMOTE_TARGET_REENTERS

        elif self.maneuver_state == ManeuverState.PROMOTE_TARGET_REENTERS:
            if self._is_lane_change_complete(self.promote_target_id):
                self._reset_lane_state(self.promote_target_id)
                self.maneuver_state = ManeuverState.REORDER_COMPLETE
                self.finalize_maneuver()

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
                if self._force_lane_change(self.exiting_leader_id, reenter_dir):
                    self._rejoin_timer = None
                    return
        self._rejoin_timer = threading.Timer(0.1, self._rejoin_tick)
        self._rejoin_timer.start()

    def finalize_maneuver(self):
        if self._rejoin_timer is not None:
            try:
                self._rejoin_timer.cancel()
            except Exception:
                pass
            self._rejoin_timer = None

        if self.exiting_leader_id != -1:
            self.platoon_state.rotate_leader_to_tail()
        elif self.promote_target_id != -1:
            self.platoon_state.promote_to_lead(self.promote_target_id)
        self._publish_truck_order()

        if self.maneuver_start_time is not None:
            self.last_maneuver_duration_sec = max(0.0, time.monotonic() - self.maneuver_start_time)
            self.maneuver_start_time = None
            self._write_scenario_log_row()

        self.platoon_state.reset_all_lane_changes()
        for truck_id in range(3):
            self._reset_lane_state(truck_id)

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

    def destroy_node(self):
        if self._scenario_log_file is not None:
            try:
                self._scenario_log_file.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TmScenarioRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
