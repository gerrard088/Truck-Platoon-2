import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32, Int32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_srvs.srv import SetBool
import queue
import os
import sys
import threading
import time
import cv2
import numpy as np
import carla
from .pid_controller import PIDController
from .lane_detect import apply_birds_eye_view, detect_lane, calculate_steering
from .command_publisher import publish_commands
from .distance_sensor import DistanceSensor
from .platooning_manager import PlatooningManager

class ManeuverState:
    IDLE = 0
    # --- 시나리오 1: Reorder (선두 -> 후미) ---
    LEADER_EXITS_LANE = 1
    LEADER_CREATES_GAP = 2
    SUCCESSOR_ENTERS_GAP = 3
    FOLLOWER_ENTERS_GAP = 4
    LEADER_REENTERS_LANE = 7
    # --- 시나리오 2: Promote (후행 -> 선두) ---
    PROMOTE_TARGET_EXITS = 21
    PROMOTE_PLATOON_CREATES_GAP = 22
    PROMOTE_TARGET_REENTERS = 23
    # --- 공통 완료/쿨다운 상태 ---
    REORDER_COMPLETE = 5
    COOLDOWN = 6

class LaneFollowingNode(Node):
    def __init__(self):
        super().__init__('lane_following_node')
        self.emergency_stop = False
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        
        # platoon order & maneuver state
        self.truck_order = [0, 1, 2]
        self.maneuver_state = ManeuverState.IDLE
        self.reorder_direction = 'left'
        
        # Maneuver-specific variables
        self.exiting_leader_id = -1
        self.successor_id = -1
        self.follower_id = -1
        self.promote_target_id = -1
        self.promote_original_leader_id = -1
        self.current_leader_id = self.truck_order[0]
        self.current_successor_id = self.truck_order[1]
        self.current_follower_id = self.truck_order[2]
        self.maneuver_start_time = None
        self.last_maneuver_duration_sec = 0.0
        self.maneuver_count = 0

        # IO wiring
        self.distance_sensor = {i: DistanceSensor(self, f'truck{i}') for i in range(3)}
        self.platooning_manager = {i: PlatooningManager(self, f'truck{i}') for i in range(3)}
        self.steer_publishers = {i: self.create_publisher(Float32, f'/truck{i}/steer_control', 10) for i in range(3)}
        self.throttle_publishers = {i: self.create_publisher(Float32, f'/truck{i}/throttle_control', 10) for i in range(3)}
        self.order_publisher = self.create_publisher(Int32MultiArray, '/platoon_order', 10)
        self.maneuver_elapsed_publisher = self.create_publisher(Float32, '/platoon_maneuver_elapsed_sec', 10)
        self.maneuver_last_duration_publisher = self.create_publisher(Float32, '/platoon_maneuver_last_duration_sec', 10)
        self.maneuver_count_publisher = self.create_publisher(Int32, '/platoon_maneuver_count', 10)
        self.velocity_subscribers = {
            i: self.create_subscription(Float32, f'/truck{i}/velocity', lambda msg, id=i: self.velocity_callback(msg, id), 10)
            for i in range(3)
        }
        self.camera_subscribers = {
            i: self.create_subscription(Image, f'/truck{i}/front_camera', lambda msg, id=i: self.camera_callback(msg, id), qos_profile)
            for i in range(3)
        }
        self.ss_subscribers = {
            i: self.create_subscription(Image, f'/truck{i}/front_camera_ss', lambda msg, id=i: self.ss_callback(msg, id), qos_profile)
            for i in range(3)
        }

        self.current_velocities = {i: 0.0 for i in range(3)}
        self.target_velocity = 19.5
        self.last_steering = {i: 0.0 for i in range(3)}
        self.last_left_fit = {i: None for i in range(3)}
        self.last_right_fit = {i: None for i in range(3)}
        self.last_left_slope = {i: 0.0 for i in range(3)}
        self.last_right_slope = {i: 0.0 for i in range(3)}
        self.last_target_waypoint = {i: None for i in range(3)}
        self.waypoint_routes = {i: [] for i in range(3)}
        self.lane_change_waypoint_routes = {i: {'left': [], 'right': []} for i in range(3)}
        self.leader_steering = 0.0
        self.bridge = CvBridge()
        self.pid_controllers = {i: PIDController(Kp=0.8, Ki=0.01, Kd=0.2) for i in range(3)}
        self.truck_views = {i: None for i in range(3)}
        self.ss_masks_bev = {i: None for i in range(3)}
        self.waypoint_tracking_only = False
        self.waypoint_route_spacing_m = 2.0
        self.waypoint_route_horizon_m = 30.0
        self.waypoint_pass_radius_m = 1.5
        self.waypoint_route_rebuild_min_m = 10.0
        self.waypoint_lookahead_min = 4.0
        self.waypoint_lookahead_max = 8.0
        self.waypoint_lookahead_gain = 0.2
        self.waypoint_lane_width_m = 3.5
        self.waypoint_cte_weight = 1.0
        self.waypoint_heading_weight = 1.2
        self.waypoint_lane_change_lookahead_bias_m = 4.0
        self.waypoint_lane_change_lookahead_min = 8.0
        self.waypoint_lane_change_lookahead_max = 14.0
        
        # lane-change FSM params
        self.current_target_lane = {i: 'center' for i in range(3)}
        self.transition_factor = {i: 0.0 for i in range(3)}
        self.transition_timer = {i: None for i in range(3)}
        
        # services & timers
        self.change_lane_service = self.create_service(SetBool, 'change_lane', self.change_lane_callback)
        self.change_queue = queue.Queue()
        self.change_timer = self.create_timer(0.1, self.process_change_queue)
        self.control_timer = self.create_timer(0.05, self.publish_commands_from_module)
        self.maneuver_timer = self.create_timer(0.2, self.manage_reorder_maneuver)
        self.maneuver_metrics_timer = self.create_timer(0.1, self._publish_maneuver_metrics)
        self.auto_scenario_timer = self.create_timer(0.1, self._maybe_trigger_auto_scenario)
        
        # lane-change timing tunables
        self.lc_dt = 0.1
        self.lc_step = 0.04
        self.lc_step_left = 0.032
        self.lc_step_right = 0.02
        
        # maneuver timing/speeds
        self.leader_slow_factor = 0.65
        self.cooldown_sec = 1.0 # 기동 후 안정화를 위해 1초 쿨다운
        
        # LiDAR Guard parameters
        self.change_min_dist = 20.0
        self.change_check_period = 0.05
        self.change_timeout_sec = 5.0 # 가드 타임아웃 5초 설정
        self.clear_start_dist = 25.0
        self.relax_full_gain = 8.0
        self.relax_full_time = 1.2
        self.relax_floor = 18.0
        self.guard_scale = {"SUCCESSOR": 0.9, "FOLLOWER": 0.9, "PROMOTE_TARGET": 1.0, "REJOIN": 1.0, "": 1.0}
        self._guard_pending = {i: False for i in range(3)}
        self._guard_timer = {}

        # Rejoin check parameters
        self._rejoin_timer = None
        self.rejoin_check_period = 0.05

        # CARLA 좌표 기반 기동 파라미터
        self.REORDER_SAFE_GAP_DISTANCE_M = 25.0 
        self.PROMOTE_SAFE_REENTRY_DISTANCE_M = 10.0
        self.REJOIN_DISTANCE_BAND_M = (10.0, 18.0)
        self.MAX_LATERAL_OFFSET_M = 1.0

        # Auto scenario trigger on a designated straight segment.
        self.auto_scenario_enabled = True
        self.auto_trigger_center_x = -9.2
        self.auto_trigger_center_y = -93.3
        self.auto_trigger_half_x_m = 3.0 + self.waypoint_lane_width_m
        self.auto_trigger_half_y_m = 12.0
        self.auto_trigger_latched = False
        self.auto_next_direction = 'left'
        
        # CARLA Actor 정보 저장
        self.carla_actors = {}
        try:
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(5.0)
            self.world = self.client.get_world()
            all_actors = self.world.get_actors()
            vehicle_actors = all_actors.filter('vehicle.*')
            for actor in vehicle_actors:
                if 'role_name' in actor.attributes:
                    role_name = actor.attributes['role_name']
                    if role_name.startswith('truck'):
                        try:
                            truck_id = int(role_name.replace('truck', ''))
                            if truck_id in range(3):
                                self.carla_actors[truck_id] = actor
                                self.get_logger().info(f"CARLA actor for '{role_name}' found and stored.")
                        except (ValueError, IndexError):
                            pass
            if len(self.carla_actors) != 3:
                self.get_logger().warn(f"Warning: Found {len(self.carla_actors)} truck actors, expected 3.")
            
            self.carla_map = self.world.get_map()
            self.get_logger().info("CARLA map successfully loaded.")

        except Exception as e:
            self.get_logger().error(f"Failed to connect to CARLA or find actors: {e}")
            self.world = None
            self.carla_map = None

        self._publish_truck_order()
        self._refresh_idle_role_assignment()
        self._publish_maneuver_metrics()

    def _publish_truck_order(self):
        msg = Int32MultiArray()
        msg.data = list(self.truck_order)
        self.order_publisher.publish(msg)

    def _refresh_idle_role_assignment(self):
        if len(self.truck_order) >= 3:
            self.current_leader_id = self.truck_order[0]
            self.current_successor_id = self.truck_order[1]
            self.current_follower_id = self.truck_order[2]
            self.get_logger().info(
                f"현재 역할 재정의 -> Leader: {self.current_leader_id}, "
                f"Follower1: {self.current_successor_id}, Follower2: {self.current_follower_id}"
            )

    def _start_maneuver_timer(self):
        self.maneuver_count += 1
        self.maneuver_start_time = time.monotonic()
        self.get_logger().info(f"현재 자동/수동 기동 회차: {self.maneuver_count}회")
        self._publish_maneuver_metrics()

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

        count_msg = Int32()
        count_msg.data = int(self.maneuver_count)
        self.maneuver_count_publisher.publish(count_msg)

    def _is_in_auto_trigger_zone(self, location) -> bool:
        return (
            abs(float(location.x) - self.auto_trigger_center_x) <= self.auto_trigger_half_x_m
            and abs(float(location.y) - self.auto_trigger_center_y) <= self.auto_trigger_half_y_m
        )

    def _maybe_trigger_auto_scenario(self):
        leader_id = self.truck_order[0] if self.truck_order else None
        leader_tf = self.get_vehicle_transform(leader_id) if leader_id is not None else None
        if leader_tf is None:
            return

        in_zone = self._is_in_auto_trigger_zone(leader_tf.location)
        if not in_zone:
            self.auto_trigger_latched = False
            return

        if self.auto_trigger_latched:
            return

        if not self.auto_scenario_enabled or self.maneuver_state != ManeuverState.IDLE:
            return

        direction = self.auto_next_direction
        if self.start_reorder_maneuver(direction):
            self.auto_trigger_latched = True
            self.auto_next_direction = 'right' if direction == 'left' else 'left'
            self.get_logger().info(
                f"Auto scenario triggered at ({leader_tf.location.x:.1f}, {leader_tf.location.y:.1f}) -> {direction}"
            )

    def get_vehicle_waypoint(self, truck_id):
        actor = self.carla_actors.get(truck_id)
        if actor and self.carla_map:
            return self.carla_map.get_waypoint(actor.get_location())
        return None

    def _distance_between_locations(self, loc_a, loc_b):
        dx = float(loc_a.x - loc_b.x)
        dy = float(loc_a.y - loc_b.y)
        dz = float(loc_a.z - loc_b.z)
        return float(np.sqrt(dx * dx + dy * dy + dz * dz))

    def _distance_between_waypoints(self, wp_a, wp_b):
        return self._distance_between_locations(wp_a.transform.location, wp_b.transform.location)

    def _heading_delta(self, yaw_a_deg, yaw_b_deg):
        delta = np.radians(yaw_b_deg - yaw_a_deg)
        return float(np.arctan2(np.sin(delta), np.cos(delta)))

    def _smooth_transition_value(self, transition_factor):
        tf = float(np.clip(transition_factor, 0.0, 1.0))
        return float(tf * tf * (3.0 - 2.0 * tf))

    def _extract_target_transform(self, target):
        if target is None:
            return None
        if hasattr(target, 'location') and hasattr(target, 'rotation'):
            return target

        transform_attr = getattr(target, 'transform', None)
        if transform_attr is None:
            return target
        if callable(transform_attr):
            try:
                transform_attr = transform_attr()
            except TypeError:
                return target
        return transform_attr

    def _blend_target_transforms(self, current_target, adjacent_target, transition_factor):
        current_tf = self._extract_target_transform(current_target)
        adjacent_tf = self._extract_target_transform(adjacent_target)
        if current_tf is None:
            return adjacent_tf
        if adjacent_tf is None:
            return current_tf

        blend = self._smooth_transition_value(transition_factor)
        location = carla.Location(
            x=float(current_tf.location.x + (adjacent_tf.location.x - current_tf.location.x) * blend),
            y=float(current_tf.location.y + (adjacent_tf.location.y - current_tf.location.y) * blend),
            z=float(current_tf.location.z + (adjacent_tf.location.z - current_tf.location.z) * blend),
        )
        yaw_delta_deg = float(np.degrees(self._heading_delta(current_tf.rotation.yaw, adjacent_tf.rotation.yaw)))
        rotation = carla.Rotation(
            pitch=float(current_tf.rotation.pitch + (adjacent_tf.rotation.pitch - current_tf.rotation.pitch) * blend),
            yaw=float(current_tf.rotation.yaw + yaw_delta_deg * blend),
            roll=float(current_tf.rotation.roll + (adjacent_tf.rotation.roll - current_tf.rotation.roll) * blend),
        )
        return carla.Transform(location, rotation)

    def _limit_steering_delta(self, truck_id, steering_angle, dt):
        prev = float(self.last_steering.get(truck_id, 0.0))
        max_delta = self.steering_slew_rate_deg_per_sec * max(1e-3, float(dt))
        return float(np.clip(steering_angle, prev - max_delta, prev + max_delta))

    def _prune_route_waypoints(self, route, transform):
        while len(route) > 1:
            first_wp = route[0]
            distance = self._distance_between_locations(transform.location, first_wp.transform.location)
            forward_dist = self._project_forward_distance(transform, first_wp.transform.location)
            if distance <= self.waypoint_pass_radius_m or forward_dist < -0.5:
                route.pop(0)
                continue
            break

    def _choose_route_target_from_route(self, route, transform, lookahead):
        if not route:
            return None

        accumulated = self._distance_between_locations(transform.location, route[0].transform.location)
        if accumulated >= lookahead:
            return route[0]

        for idx in range(1, len(route)):
            accumulated += self._distance_between_waypoints(route[idx - 1], route[idx])
            if accumulated >= lookahead:
                return route[idx]

        return route[-1]

    def _project_forward_distance(self, transform, target_location):
        vehicle_loc = transform.location
        dx = float(target_location.x - vehicle_loc.x)
        dy = float(target_location.y - vehicle_loc.y)
        dz = float(target_location.z - vehicle_loc.z)
        forward = transform.get_forward_vector()
        return float(dx * forward.x + dy * forward.y + dz * forward.z)

    def _select_best_waypoint_vector(self, current_transform, candidates):
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        curr_loc = current_transform.location
        curr_fwd = current_transform.get_forward_vector()
        curr_fwd_2d = np.array([curr_fwd.x, curr_fwd.y])
        curr_fwd_2d /= np.linalg.norm(curr_fwd_2d)

        best_wp = None
        max_dot = -2.0

        for wp in candidates:
            target_loc = wp.transform.location
            vec_to_wp = np.array([
                target_loc.x - curr_loc.x,
                target_loc.y - curr_loc.y
            ])
            norm = np.linalg.norm(vec_to_wp)
            if norm < 0.1:
                wp_fwd = wp.transform.get_forward_vector()
                vec_to_wp = np.array([wp_fwd.x, wp_fwd.y])
            else:
                vec_to_wp /= norm
            
            dot = np.dot(curr_fwd_2d, vec_to_wp)
            if dot > max_dot:
                max_dot = dot
                best_wp = wp
        
        return best_wp

    def _find_adjacent_driving_lane(self, waypoint, direction):
        if waypoint is None:
            return None

        get_neighbor = waypoint.get_left_lane if direction == 'left' else waypoint.get_right_lane
        candidate = get_neighbor()
        visited = set()
        while candidate is not None:
            lane_ref = (candidate.road_id, candidate.section_id, candidate.lane_id)
            if lane_ref in visited:
                break
            visited.add(lane_ref)
            if candidate.lane_type == carla.LaneType.Driving:
                return candidate
            candidate = candidate.get_left_lane() if direction == 'left' else candidate.get_right_lane()
        return None

    def _build_waypoint_route(self, truck_id, seed_waypoint):
        transform = self.get_vehicle_transform(truck_id)
        if seed_waypoint is None or transform is None:
            return []

        route = [seed_waypoint]
        traveled = 0.0
        current_wp = seed_waypoint
        max_steps = max(1, int(np.ceil(self.waypoint_horizon_m / max(0.5, self.waypoint_route_spacing_m)))) + 2 if hasattr(self, 'waypoint_horizon_m') else 20

        for _ in range(max_steps):
            try:
                candidates = current_wp.next(self.waypoint_route_spacing_m)
            except Exception:
                break

            if not candidates:
                break

            next_wp = self._select_best_waypoint_vector(transform, candidates)
            if next_wp is None:
                break

            route.append(next_wp)
            traveled += self._distance_between_waypoints(current_wp, next_wp)
            current_wp = next_wp
            transform = next_wp.transform

            if traveled >= self.waypoint_route_horizon_m:
                break

        return route

    def _prune_passed_waypoints(self, truck_id, transform):
        route = self.waypoint_routes.get(truck_id, [])
        self._prune_route_waypoints(route, transform)

    def _choose_route_target(self, truck_id, transform, lookahead):
        route = self.waypoint_routes.get(truck_id, [])
        return self._choose_route_target_from_route(route, transform, lookahead)

    def _handoff_waypoint_route_to_target_lane(self, truck_id, direction):
        current_wp = self.get_vehicle_waypoint(truck_id)
        if current_wp is None:
            return False

        target_wp = self._find_adjacent_driving_lane(current_wp, direction)
        if target_wp is None:
            self.get_logger().warn(f"Truck {truck_id}: {direction} 방향 인접 차선 waypoint를 찾지 못했습니다")
            return False

        target_route = self._build_waypoint_route(truck_id, target_wp)
        if not target_route:
            self.get_logger().warn(f"Truck {truck_id}: {direction} 방향 목표 waypoint route 생성 실패")
            return False

        self.waypoint_routes[truck_id] = target_route
        self.last_target_waypoint[truck_id] = target_route[0]
        self.get_logger().info(f"Truck {truck_id}: {direction} 차선 waypoint route로 추종 경로 전환")
        return True


    def _resolve_target_waypoint(self, truck_id):
        transform = self.get_vehicle_transform(truck_id)
        current_wp = self.get_vehicle_waypoint(truck_id)
        if transform is None or current_wp is None:
            return None

        target_lane = self.current_target_lane.get(truck_id, 'center')
        is_transitioning = target_lane in ('left', 'right') and self.transition_factor.get(truck_id, 0.0) < 1.0
        route = self.waypoint_routes.get(truck_id, [])
        needs_rebuild = (
            not route
            or self._distance_between_locations(transform.location, route[-1].transform.location) < self.waypoint_route_rebuild_min_m
        )
        if needs_rebuild:
            self.waypoint_routes[truck_id] = self._build_waypoint_route(truck_id, current_wp)

        self._prune_passed_waypoints(truck_id, transform)

        speed = float(self.current_velocities.get(truck_id, 0.0))
        base_lookahead = float(np.clip(
            self.waypoint_lookahead_min + self.waypoint_lookahead_gain * speed,
            self.waypoint_lookahead_min,
            self.waypoint_lookahead_max,
        ))
        lookahead = base_lookahead
        if is_transitioning:
            lookahead = float(np.clip(
                base_lookahead + self.waypoint_lane_change_lookahead_bias_m,
                self.waypoint_lane_change_lookahead_min,
                self.waypoint_lane_change_lookahead_max,
            ))
        target_wp = self._choose_route_target(truck_id, transform, lookahead)

        if target_lane in ('left', 'right'):
            adjacent = self._find_adjacent_driving_lane(current_wp, target_lane)
            adjacent_routes = self.lane_change_waypoint_routes.get(truck_id, {})
            adjacent_route = adjacent_routes.get(target_lane, [])
            if adjacent is not None:
                needs_adjacent_rebuild = (
                    not adjacent_route
                    or self._distance_between_locations(transform.location, adjacent_route[-1].transform.location) < self.waypoint_route_rebuild_min_m
                )
                if needs_adjacent_rebuild:
                    adjacent_route = self._build_waypoint_route(truck_id, adjacent)
                    adjacent_routes[target_lane] = adjacent_route
                self._prune_route_waypoints(adjacent_route, transform)
                adjacent_target = self._choose_route_target_from_route(adjacent_route, transform, lookahead)
                if adjacent_target is not None:
                    if is_transitioning:
                        target_wp = self._blend_target_transforms(
                            target_wp,
                            adjacent_target,
                            self.transition_factor.get(truck_id, 0.0),
                        )
                    else:
                        target_wp = adjacent_target

        self.last_target_waypoint[truck_id] = target_wp
        return target_wp

    def _compute_waypoint_tracking_error(self, truck_id, target_waypoint):
        transform = self.get_vehicle_transform(truck_id)
        target_transform = self._extract_target_transform(target_waypoint)
        if transform is None or target_transform is None:
            return None

        vehicle_loc = transform.location
        target_loc = target_transform.location
        target_yaw_rad = np.radians(target_transform.rotation.yaw)

        dx = target_loc.x - vehicle_loc.x
        dy = target_loc.y - vehicle_loc.y
        right_x = -np.sin(target_yaw_rad)
        right_y = np.cos(target_yaw_rad)
        lateral_error_m = dx * right_x + dy * right_y
        heading_error = self._heading_delta(transform.rotation.yaw, target_transform.rotation.yaw)

        normalized_cte = lateral_error_m / max(1e-3, self.waypoint_lane_width_m)
        return float((self.waypoint_cte_weight * normalized_cte) + (self.waypoint_heading_weight * heading_error))

    def _cancel_guard(self, truck_id, tag):
        key = (truck_id, tag)
        t = self._guard_timer.pop(key, None)
        if t:
            try: t.cancel()
            except Exception: pass
        self._guard_pending[truck_id] = False

    def _guarded_lane_change(self, truck_id, direction, tag=''):
        if self._guard_pending.get(truck_id, False):
            self.get_logger().info(f"[{tag}] Truck {truck_id} 가드 이미 진행 중")
            return
        self._guard_pending[truck_id] = True
        start_ts = time.monotonic()
        scale = self.guard_scale.get(tag, 1.0)
        def _check_and_change():
            if tag != 'MANUAL' and self.maneuver_state in (ManeuverState.IDLE, ManeuverState.REORDER_COMPLETE, ManeuverState.COOLDOWN):
                self.get_logger().info(f"[{tag}] Truck {truck_id} 가드 취소(FSM 종료/전이)")
                return self._cancel_guard(truck_id, tag)
            if (self.change_timeout_sec is not None) and (time.monotonic() - start_ts > self.change_timeout_sec):
                self.get_logger().warn(f"[{tag}] Truck {truck_id} 타임아웃({self.change_timeout_sec}s). 취소.")
                return self._cancel_guard(truck_id, tag)
            elapsed = time.monotonic() - start_ts
            relax_ratio = min(1.0, elapsed / max(1e-3, self.relax_full_time))
            relax_gain  = self.relax_full_gain * relax_ratio
            dyn_min = max(self.relax_floor, (self.change_min_dist - relax_gain)) * scale
            dist = self.distance_sensor[truck_id].get_distance()
            if dist is None or dist > self.clear_start_dist:
                self.get_logger().info(f"[{tag}] Truck {truck_id} FAST-START: dist={dist}m >= clear={self.clear_start_dist:.1f}m")
                self._cancel_guard(truck_id, tag)
                return self.change_lane(truck_id, direction)
            if dist >= dyn_min:
                self.get_logger().info(f"[{tag}] Truck {truck_id} OK: dist={dist:.1f}m >= dyn_min={dyn_min:.1f}m → 차선 변경 시작")
                self._cancel_guard(truck_id, tag)
                return self.change_lane(truck_id, direction)
            dstr = "None" if dist is None else f"{dist:.1f}"
            self.get_logger().info(f"[{tag}] Truck {truck_id} 대기: dist={dstr}m (< {dyn_min:.1f}m). 재확인 예정")
            key = (truck_id, tag)
            timer = threading.Timer(self.change_check_period, _check_and_change)
            self._guard_timer[key] = timer
            timer.start()
        _check_and_change()
    
    def get_vehicle_transform(self, truck_id):
        actor = self.carla_actors.get(truck_id)
        if actor:
            try:
                return actor.get_transform()
            except Exception as e:
                self.get_logger().warn(f"Could not get transform for truck {truck_id}: {e}")
                return None
        return None

    def _distance_to_platoon_leader(self, follower_id):
        if follower_id not in self.truck_order:
            return None
        follower_rank = self.truck_order.index(follower_id)
        if follower_rank <= 0:
            return None

        leader_id = self.truck_order[follower_rank - 1]
        follower_actor = self.carla_actors.get(follower_id)
        leader_actor = self.carla_actors.get(leader_id)
        follower_tf = self.get_vehicle_transform(follower_id)
        leader_tf = self.get_vehicle_transform(leader_id)
        if follower_actor is None or leader_actor is None or follower_tf is None or leader_tf is None:
            return None

        relative_vec = leader_tf.location - follower_tf.location
        follower_forward = follower_tf.get_forward_vector()
        forward_distance = float(relative_vec.dot(follower_forward))
        if forward_distance <= 0.0:
            return None

        follower_extent = float(getattr(follower_actor.bounding_box.extent, 'x', 0.0))
        leader_extent = float(getattr(leader_actor.bounding_box.extent, 'x', 0.0))
        gap_distance = forward_distance - follower_extent - leader_extent
        return max(0.0, gap_distance)

    def ss_callback(self, msg: Image, truck_id: int):
        try:
            ss_img = self.bridge.imgmsg_to_cv2(msg, 'mono8')
            lane_mask = (ss_img == 24).astype(np.uint8) * 255
            lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
            mask_bgr = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
            mask_bev = apply_birds_eye_view(mask_bgr)
            mask_bev = cv2.cvtColor(mask_bev, cv2.COLOR_BGR2GRAY)
            self.ss_masks_bev[truck_id] = mask_bev
        except Exception as e:
            self.get_logger().error(f"SS callback error (truck {truck_id}): {e}")

    def velocity_callback(self, msg, truck_id):
        self.current_velocities[truck_id] = msg.data

    def _maybe_complete_camera_lane_change(self, truck_id, lane_positions, img_width, steering_angle):
        target_lane = self.current_target_lane.get(truck_id, 'center')
        if target_lane == 'center':
            return False
        if self.transition_factor.get(truck_id, 0.0) < 0.55:
            return False
        if not lane_positions:
            return False

        center_left = lane_positions.get('center_left')
        center_right = lane_positions.get('center_right')
        if center_left is None or center_right is None:
            return False

        lane_center = (center_left + center_right) / 2.0
        img_center = img_width / 2.0
        normalized_center_error = abs(lane_center - img_center) / max(1.0, img_center)
        steering_abs = abs(float(steering_angle))

        if normalized_center_error <= 0.06 and steering_abs <= 4.0:
            self.get_logger().info(
                f"Truck {truck_id}: 카메라 기준 차선 변경 조기 완료 \
(center_error={normalized_center_error:.3f}, steer={steering_abs:.2f})"
            )
            self.reset_lane_state(truck_id)
            return True
        return False

    def camera_callback(self, msg, truck_id):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        bev_image = apply_birds_eye_view(frame)
        ss_mask_bev = self.ss_masks_bev.get(truck_id)
        result = detect_lane(
            bev_image, truck_id,
            self.last_left_fit[truck_id],
            self.last_right_fit[truck_id],
            self.last_left_slope[truck_id],
            self.last_right_slope[truck_id],
            ss_mask=ss_mask_bev
        )
        lane_image, lane_positions, lfit, rfit, lslope, rslope = self._normalize_detect_lane_output(result, truck_id)
        self.last_left_fit[truck_id] = lfit
        self.last_right_fit[truck_id] = rfit
        self.last_left_slope[truck_id] = lslope
        self.last_right_slope[truck_id] = rslope
        is_changing = self.current_target_lane[truck_id] != 'center'
        is_transitioning = is_changing and self.transition_factor[truck_id] < 1.0
        steering_angle = 0.0
        used_waypoint = False

        if not is_changing:
            target_waypoint = self._resolve_target_waypoint(truck_id)
            waypoint_error = self._compute_waypoint_tracking_error(truck_id, target_waypoint)
            if waypoint_error is not None:
                pid_output = self.pid_controllers[truck_id].compute(waypoint_error)
                steering_angle = float(np.clip(-pid_output * 80.0, -70.0, 70.0))
                used_waypoint = True

        if not used_waypoint and not self.waypoint_tracking_only:
            steering_angle = calculate_steering(
                self.pid_controllers[truck_id],
                lane_positions,
                frame.shape[1],
                self.current_target_lane[truck_id],
                self.transition_factor[truck_id],
                is_lane_changing=is_transitioning
            )

        self._maybe_complete_camera_lane_change(
            truck_id, lane_positions, frame.shape[1], steering_angle
        )

        steer_msg = Float32(); steer_msg.data = steering_angle
        self.steer_publishers[truck_id].publish(steer_msg)
        self.last_steering[truck_id] = steering_angle
        if truck_id == self.truck_order[0]:
            self.leader_steering = steering_angle
        self.truck_views[truck_id] = lane_image

    def _normalize_detect_lane_output(self, result, truck_id):
        lfit = self.last_left_fit[truck_id]
        rfit = self.last_right_fit[truck_id]
        lslope = self.last_left_slope[truck_id]
        rslope = self.last_right_slope[truck_id]
        if not isinstance(result, (list, tuple)) or len(result) < 2:
            raise RuntimeError("detect_lane return format invalid")
        lane_image, lane_positions = result[0], result[1]
        if len(result) >= 6:
            return lane_image, lane_positions, result[2], result[3], result[4], result[5]
        if len(result) == 3:
            extra = result[2]
            if isinstance(extra, dict):
                lfit = extra.get('left_fit', lfit)
                rfit = extra.get('right_fit', rfit)
                lslope = extra.get('left_slope', lslope)
                rslope = extra.get('right_slope', rslope)
            elif isinstance(extra, (list, tuple)) and len(extra) >= 4:
                lfit, rfit, lslope, rslope = extra[0], extra[1], extra[2], extra[3]
        return lane_image, lane_positions, lfit, rfit, lslope, rslope

    def process_change_queue(self):
        try:
            while not self.change_queue.empty():
                item = self.change_queue.get_nowait()
                if isinstance(item, tuple) and len(item) == 3 and item[0] == 'promote':
                    _, target_id, direction = item
                    self.get_logger().info(f"[Keyboard] Promote 요청 (타겟: {target_id}, 방향: {direction})")
                    self.start_promote_maneuver(target_id, direction)
                elif isinstance(item, tuple) and len(item) == 2:
                    command, direction = item
                    if command == 'reorder':
                        self.get_logger().info(f"[Keyboard] 순서 재배치 요청 (방향: {direction})")
                        self.start_reorder_maneuver(direction)
                    else:
                        truck_id = command
                        if self.maneuver_state == ManeuverState.IDLE:
                            self.get_logger().info(f"[Keyboard] Truck {truck_id} '{direction}' 차선 변경 요청")
                            self._guarded_lane_change(truck_id, direction, tag="MANUAL")
                        else:
                            self.get_logger().warn("기동 중에는 개별 차선 변경을 허용하지 않습니다.")
        except queue.Empty:
            pass
        except Exception as e:
            self.get_logger().error(f"큐 처리 중 에러: {e}")

    def change_lane_callback(self, request, response):
        direction = 'left' if request.data else 'right'
        if self.maneuver_state != ManeuverState.IDLE:
            response.success = False
            response.message = "재배치 중에는 서비스로 개별 변경 불가"
            return response
        self.get_logger().info(f"서비스 호출: 리더부터 {direction} 방향 순차 차선 변경")
        if self._guarded_lane_change(self.truck_order[0], direction):
            followers = [tid for tid in self.truck_order if tid != self.truck_order[0]]
            for i, follower_id in enumerate(followers):
                threading.Timer((i + 1) * 2.0, lambda: self._guarded_lane_change(follower_id, direction)).start()
            response.success = True
            response.message = f"{direction} 차선 변경 절차 시작됨"
        else:
            response.success = False
            response.message = "차선 변경 실패"
        return response

    def change_lane(self, truck_id, direction):
        if self.current_target_lane.get(truck_id, 'center') != 'center':
            self.get_logger().warn(f"Truck {truck_id}: 이미 차선 변경 중")
            return False
        self.get_logger().info(f"Truck {truck_id}: {direction}으로 차선 변경 시작")
        self.current_target_lane[truck_id] = direction
        self.pid_controllers[truck_id].reset()
        self.transition_factor[truck_id] = 0.0
        self._handoff_waypoint_route_to_target_lane(truck_id, direction)
        transition_step = self.lc_step_left if direction == 'left' else self.lc_step_right
        self.get_logger().info(
            f"Truck {truck_id}: {direction} 차선 변경 전환 속도 step={transition_step:.3f}"
        )

        def update_transition():
            if self.current_target_lane.get(truck_id) != direction:
                if self.transition_timer.get(truck_id):
                    self.transition_timer[truck_id].cancel()
                    self.transition_timer[truck_id] = None
                return

            self.transition_factor[truck_id] += transition_step
            if self.transition_factor[truck_id] >= 1.0:
                self.transition_factor[truck_id] = 1.0
                self.get_logger().info(f"Truck {truck_id}: 차선 변경 애니메이션 완료")
                if self.transition_timer.get(truck_id):
                    self.transition_timer[truck_id].cancel()
                    self.transition_timer[truck_id] = None
        
        if self.transition_timer.get(truck_id):
            self.transition_timer[truck_id].cancel()
        self.transition_timer[truck_id] = self.create_timer(self.lc_dt, update_transition)
        return True

    def reset_lane_state(self, truck_id):
        self.get_logger().info(f"Truck {truck_id}: 중앙 차선 복귀 상태로 리셋")
        if self.transition_timer.get(truck_id):
            self.transition_timer[truck_id].cancel()
            self.transition_timer[truck_id] = None
        self.current_target_lane[truck_id] = 'center'
        self.transition_factor[truck_id] = 1.0
        self.pid_controllers[truck_id].reset()
        self.last_left_fit[truck_id] = None
        self.last_right_fit[truck_id] = None
        self.last_target_waypoint[truck_id] = None
        self.waypoint_routes[truck_id] = []
        self.lane_change_waypoint_routes[truck_id] = {'left': [], 'right': []}

    def start_reorder_maneuver(self, direction):
        if self.maneuver_state != ManeuverState.IDLE:
            self.get_logger().warn("이미 재배치 진행 중")
            return False
        self.reorder_direction = direction
        self.exiting_leader_id = self.truck_order[0]
        self.successor_id = self.truck_order[1]
        self.follower_id = self.truck_order[2]
        self.maneuver_state = ManeuverState.LEADER_EXITS_LANE
        self._start_maneuver_timer()
        self.get_logger().info(f"= 재배치 시작({direction}) : 리더 {self.exiting_leader_id} 차선이탈 =")
        self.change_lane(self.exiting_leader_id, self.reorder_direction)
        return True

    def start_promote_maneuver(self, target_id, direction):
        if self.maneuver_state != ManeuverState.IDLE:
            self.get_logger().warn("이미 다른 기동 진행 중")
            return False
        if target_id not in self.truck_order[1:]:
            self.get_logger().warn(f"Truck {target_id}는 후행 차량이 아니므로 선두로 보낼 수 없습니다.")
            return False

        self.reorder_direction = direction
        self.promote_target_id = target_id
        self.promote_original_leader_id = self.truck_order[0]
        self.maneuver_state = ManeuverState.PROMOTE_TARGET_EXITS
        self._start_maneuver_timer()
        self.get_logger().info(f"= Promote 시작({direction}) : 타겟 차량 {self.promote_target_id} 차선 이탈 =")
        self.change_lane(self.promote_target_id, self.reorder_direction)
        return True

    def manage_reorder_maneuver(self):
        if self.maneuver_state == ManeuverState.IDLE:
            return
        
        def is_lane_change_complete(truck_id):
            return self.transition_factor.get(truck_id, 0.0) >= 1.0

        # --- 시나리오 1: Reorder (선두 -> 후미) ---
        if self.maneuver_state == ManeuverState.LEADER_EXITS_LANE:
            if is_lane_change_complete(self.exiting_leader_id):
                self.reset_lane_state(self.exiting_leader_id)
                self.maneuver_state = ManeuverState.LEADER_CREATES_GAP
                self.get_logger().info("리더 갭 생성 단계 진입(감속 및 거리 확인 시작)")
        
        # [수정된 최종 로직]
        elif self.maneuver_state == ManeuverState.LEADER_CREATES_GAP:
            leader_tf = self.get_vehicle_transform(self.exiting_leader_id)
            successor_tf = self.get_vehicle_transform(self.successor_id)

            if leader_tf and successor_tf:
                # 후속 차량(successor)의 전방 벡터를 기준으로 리더의 상대 위치 계산
                relative_vec = leader_tf.location - successor_tf.location
                successor_forward_vec = successor_tf.get_forward_vector()
                
                # forward_dist: 리더가 후속 차량보다 뒤에 있으면 음수, 앞에 있으면 양수
                forward_dist = relative_vec.dot(successor_forward_vec)
                
                is_behind = forward_dist < 0
                longitudinal_distance = abs(forward_dist) # 종방향 거리

                # 조건: 1. 리더가 후속 차량 뒤에 있고, 2. 그 거리가 안전거리 이상일 때
                if is_behind and longitudinal_distance >= self.REORDER_SAFE_GAP_DISTANCE_M:
                    self.get_logger().info(
                        f"안전거리 확보 (뒤로 {longitudinal_distance:.1f}m). 후속 차량 진입 시작."
                    )
                    self.maneuver_state = ManeuverState.SUCCESSOR_ENTERS_GAP
                    self._guarded_lane_change(self.successor_id, self.reorder_direction, tag="SUCCESSOR")
                else:
                    # 현재 상태 로깅 (디버깅에 유용)
                    self.get_logger().debug(
                        f"갭 생성 대기 중... 리더 상대 위치: {forward_dist:.1f}m "
                        f"(목표: 뒤로 {self.REORDER_SAFE_GAP_DISTANCE_M}m 이상)"
                    )
            else:
                 self.get_logger().debug("Reorder: 차량 transform 정보를 기다리는 중...")

        elif self.maneuver_state == ManeuverState.SUCCESSOR_ENTERS_GAP:
            if is_lane_change_complete(self.successor_id):
                self.reset_lane_state(self.successor_id)
                self.maneuver_state = ManeuverState.FOLLOWER_ENTERS_GAP
                self.get_logger().info("후속 차량(2번) 차선 변경 시도(라이다 가드)")
                self._guarded_lane_change(self.follower_id, self.reorder_direction, tag="FOLLOWER")

        elif self.maneuver_state == ManeuverState.FOLLOWER_ENTERS_GAP:
            if is_lane_change_complete(self.follower_id):
                self.reset_lane_state(self.follower_id)
                self.maneuver_state = ManeuverState.LEADER_REENTERS_LANE
                self.get_logger().info("리더 재합류 대기 시작")
                self._leader_reenter()

        elif self.maneuver_state == ManeuverState.LEADER_REENTERS_LANE:
            if is_lane_change_complete(self.exiting_leader_id):
                self.maneuver_state = ManeuverState.REORDER_COMPLETE
                self.finalize_reorder()
        
        # --- 시나리오 2: Promote (후행 -> 선두) ---
        elif self.maneuver_state == ManeuverState.PROMOTE_TARGET_EXITS:
            if is_lane_change_complete(self.promote_target_id):
                self.reset_lane_state(self.promote_target_id)
                self.maneuver_state = ManeuverState.PROMOTE_PLATOON_CREATES_GAP
                self.get_logger().info("Promote: 대상 차량 이탈 완료. 군집 감속 및 추월 대기.")
        
        elif self.maneuver_state == ManeuverState.PROMOTE_PLATOON_CREATES_GAP:
            target_tf = self.get_vehicle_transform(self.promote_target_id)
            leader_tf = self.get_vehicle_transform(self.promote_original_leader_id)

            if target_tf and leader_tf:
                leader_forward_vec = leader_tf.get_forward_vector()
                relative_vec = target_tf.location - leader_tf.location
                forward_distance = relative_vec.dot(leader_forward_vec)
                
                is_ahead = forward_distance > 0
                is_safe_distance = forward_distance >= self.PROMOTE_SAFE_REENTRY_DISTANCE_M

                if is_ahead and is_safe_distance:
                    self.get_logger().info(f"Promote: 안전거리({forward_distance:.1f}m) 확보. 대상 차량 선두 합류 시도.")
                    self.maneuver_state = ManeuverState.PROMOTE_TARGET_REENTERS
                    self._promote_target_reenter()
            else:
                self.get_logger().debug("Promote: 차량 transform 정보를 기다리는 중...")

        elif self.maneuver_state == ManeuverState.PROMOTE_TARGET_REENTERS:
            if is_lane_change_complete(self.promote_target_id):
                self.maneuver_state = ManeuverState.REORDER_COMPLETE
                self.finalize_reorder()

    def _leader_reenter(self):
        if self.maneuver_state == ManeuverState.LEADER_REENTERS_LANE:
            self.get_logger().info("리더 재합류 준비: CARLA 좌표 기반 확인 루프 시작")
            self._start_rejoin_check()

    def _start_rejoin_check(self):
        if self._rejoin_timer:
            try: self._rejoin_timer.cancel()
            except Exception: pass
        self._rejoin_tick()

    def _rejoin_tick(self):
        if self.maneuver_state != ManeuverState.LEADER_REENTERS_LANE:
            self._rejoin_timer = None
            return

        leader_id = self.exiting_leader_id
        follower_id = self.follower_id
        if leader_id == -1 or follower_id == -1:
            self._rejoin_timer = None
            return

        leader_tf = self.get_vehicle_transform(leader_id)
        follower_tf = self.get_vehicle_transform(follower_id)

        if leader_tf and follower_tf:
            relative_vec = leader_tf.location - follower_tf.location
            follower_forward_vec = follower_tf.get_forward_vector()
            follower_right_vec = follower_tf.get_right_vector()
            forward_dist = relative_vec.dot(follower_forward_vec)
            lateral_dist = abs(relative_vec.dot(follower_right_vec))

            is_behind = forward_dist < 0
            is_in_longitudinal_band = self.REJOIN_DISTANCE_BAND_M[0] <= abs(forward_dist) <= self.REJOIN_DISTANCE_BAND_M[1]
            is_laterally_aligned = lateral_dist <= self.MAX_LATERAL_OFFSET_M

            if is_behind and is_in_longitudinal_band and is_laterally_aligned:
                reenter_dir = self._opposite(self.reorder_direction)
                self._guarded_lane_change(leader_id, reenter_dir, tag="REJOIN")
                self._rejoin_timer = None
                return

        self._rejoin_timer = threading.Timer(self.rejoin_check_period, self._rejoin_tick)
        self._rejoin_timer.start()

    def _promote_target_reenter(self):
        reenter_dir = self._opposite(self.reorder_direction)
        self.get_logger().info(f"Promote: 타겟 차량 {self.promote_target_id} 선두 재합류 시작 ({reenter_dir})")
        self.change_lane(self.promote_target_id, reenter_dir)

    def finalize_reorder(self):
        if self.maneuver_state == ManeuverState.COOLDOWN: return
        self.get_logger().info("===== 기동 완료: 모든 상태 초기화 =====")
        for i in range(3):
            self._cancel_guard(i, "SUCCESSOR")
            self._cancel_guard(i, "FOLLOWER")
            self._cancel_guard(i, "PROMOTE_TARGET")
            self._cancel_guard(i, "REJOIN")

        if self._rejoin_timer:
            try: self._rejoin_timer.cancel()
            except Exception: pass
            self._rejoin_timer = None

        if self.exiting_leader_id != -1:
            self.truck_order = [self.successor_id, self.follower_id, self.exiting_leader_id]
            self.get_logger().info(f"= Reorder 완료: 새로운 순서는 {self.truck_order} =")
        elif self.promote_target_id != -1:
            old_order = list(self.truck_order)
            old_order.remove(self.promote_target_id)
            self.truck_order = [self.promote_target_id] + old_order
            self.get_logger().info(f"= Promote 완료: 새로운 순서는 {self.truck_order} =")

        self._publish_truck_order()
        self._refresh_idle_role_assignment()

        if self.maneuver_start_time is not None:
            self.last_maneuver_duration_sec = max(0.0, time.monotonic() - self.maneuver_start_time)
            self.maneuver_start_time = None
            self.get_logger().info(f"기동 완료 시간: {self.last_maneuver_duration_sec:.2f}s")
            self._publish_maneuver_metrics()

        for i in range(3):
            self.reset_lane_state(i)
            pm = self.platooning_manager[i]
            pm.integral = 0.0
            pm.prev_error = 0.0
            pm.prev_time = time.time()

        self.maneuver_state = ManeuverState.COOLDOWN
        threading.Timer(self.cooldown_sec, self.reset_maneuver_variables).start()

    def _opposite(self, direction: str) -> str:
        return 'right' if direction == 'left' else 'left'

    def reset_maneuver_variables(self):
        self.maneuver_state = ManeuverState.IDLE
        self.exiting_leader_id = -1
        self.successor_id = -1
        self.follower_id = -1
        self.promote_target_id = -1
        self.promote_original_leader_id = -1
        self.get_logger().info("===== 쿨다운 종료: IDLE 상태로 복귀 =====")
        self._refresh_idle_role_assignment()
        self._publish_maneuver_metrics()

    def publish_commands_from_module(self):
        if not self.truck_order: return
        current_leader_id = self.truck_order[0]
        
        is_promoting = self.maneuver_state in (
            ManeuverState.PROMOTE_TARGET_EXITS,
            ManeuverState.PROMOTE_PLATOON_CREATES_GAP,
            ManeuverState.PROMOTE_TARGET_REENTERS
        )
        
        leader_front_distance = self.distance_sensor[current_leader_id].get_distance()
        self.emergency_stop = False
        if leader_front_distance is not None and leader_front_distance < 3.0:
            self.emergency_stop = True
            self.get_logger().warn(f"[리더 Truck {current_leader_id}] 비상 정지! 전방 장애물 거리: {leader_front_distance:.2f} m")

        for i, truck_id in enumerate(self.truck_order):
            if self.emergency_stop:
                throttle_msg = Float32(); throttle_msg.data = -1.0
                self.throttle_publishers[truck_id].publish(throttle_msg)
                continue

            if is_promoting:
                if truck_id == self.promote_target_id:
                    try:
                        rank = self.truck_order.index(self.promote_target_id)
                    except ValueError:
                        rank = -1

                    if rank == 1:
                        promote_speed = self.target_velocity * 1.2
                    elif rank == 2:
                        promote_speed = self.target_velocity * 1.1
                    else:
                        promote_speed = self.target_velocity * 1.2

                    publish_commands(
                        [self.throttle_publishers[truck_id]],
                        [self.current_velocities.get(truck_id, 0.0)],
                        promote_speed,
                        [self.last_steering.get(truck_id, 0.0)]
                    )
                    continue
                else:
                    if (self.promote_target_id == self.truck_order[1]) and (truck_id == self.truck_order[2]):
                        publish_commands(
                            [self.throttle_publishers[truck_id]],
                            [self.current_velocities.get(truck_id, 0.0)],
                            self.target_velocity,
                            [self.last_steering.get(truck_id, 0.0)]
                        )
                        continue

                if self.maneuver_state in (ManeuverState.PROMOTE_PLATOON_CREATES_GAP,
                                           ManeuverState.PROMOTE_TARGET_REENTERS):
                    gap_creation_speed = self.target_velocity * 0.8
                    publish_commands(
                        [self.throttle_publishers[truck_id]],
                        [self.current_velocities.get(truck_id, 0.0)],
                        gap_creation_speed,
                        [self.last_steering.get(truck_id, 0.0)]
                    )
                    continue

            if i == 0: # Current Leader
                final_target_velocity = self.target_velocity
                if leader_front_distance is not None:
                    if leader_front_distance <= 10.0: final_target_velocity = 0.0
                    elif leader_front_distance < 30.0:
                        ratio = (leader_front_distance - 10.0) / 20.0
                        final_target_velocity = max(0.0, self.target_velocity * ratio)
                
                slow_phases = (
                    ManeuverState.LEADER_CREATES_GAP,
                )
                if truck_id == self.exiting_leader_id and self.maneuver_state in slow_phases:
                    gap_creation_speed = self.target_velocity * self.leader_slow_factor
                    final_target_velocity = min(final_target_velocity, gap_creation_speed)

                publish_commands(
                    [self.throttle_publishers[truck_id]],
                    [self.current_velocities.get(truck_id, 0.0)],
                    final_target_velocity,
                    [self.last_steering.get(truck_id, 0.0)]
                )
            else: # Followers
                lidar_distance = self.distance_sensor[truck_id].get_distance()
                LOSS_DIST = 40.0
                if lidar_distance is None or lidar_distance > LOSS_DIST:
                    publish_commands(
                        [self.throttle_publishers[truck_id]],
                        [self.current_velocities.get(truck_id, 0.0)],
                        self.target_velocity,
                        [self.last_steering.get(truck_id, 0.0)]
                    )
                else:
                    leader_id = self.truck_order[i - 1]
                    leader_vel = self.current_velocities.get(leader_id, self.target_velocity)
                    ego_vel = self.current_velocities.get(truck_id, 0.0)
                    self.platooning_manager[truck_id].update_distance(
                        lidar_distance,
                        leader_vel,
                        emergency_stop=self.emergency_stop,
                        ego_velocity=ego_vel,
                    )

def opencv_loop(node: LaneFollowingNode):
    window_name = "Combined Bird-Eye View"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    while rclpy.ok():
        views = [node.truck_views.get(i) if node.truck_views.get(i) is not None else np.zeros((480, 640, 3), dtype=np.uint8) for i in node.truck_order]
        if len(views) == 3: cv2.imshow(window_name, cv2.hconcat(views))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('t'): node.change_queue.put(('reorder', 'left'))
        elif key == ord('y'): node.change_queue.put(('reorder', 'right'))
        elif key == ord('j'):
            if len(node.truck_order) > 1: node.change_queue.put(('promote', node.truck_order[1], 'left'))
        elif key == ord('k'):
            if len(node.truck_order) > 2: node.change_queue.put(('promote', node.truck_order[2], 'left'))
        elif key == 27: break
    cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowingNode()
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()
    try:
        opencv_loop(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
