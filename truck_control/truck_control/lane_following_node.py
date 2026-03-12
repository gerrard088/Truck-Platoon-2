import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_srvs.srv import SetBool
import queue
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

        # IO wiring
        self.distance_sensor = {i: DistanceSensor(self, f'truck{i}') for i in range(3)}
        self.platooning_manager = {i: PlatooningManager(self, f'truck{i}') for i in range(3)}
        self.steer_publishers = {i: self.create_publisher(Float32, f'/truck{i}/steer_control', 10) for i in range(3)}
        self.throttle_publishers = {i: self.create_publisher(Float32, f'/truck{i}/throttle_control', 10) for i in range(3)}
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
        
        # runtime state
        self.current_velocities = {i: 0.0 for i in range(3)}
        self.target_velocity = 19.5
        self.last_steering = {i: 0.0 for i in range(3)}
        self.last_left_fit = {i: None for i in range(3)}
        self.last_right_fit = {i: None for i in range(3)}
        self.last_left_slope = {i: 0.0 for i in range(3)}
        self.last_right_slope = {i: 0.0 for i in range(3)}
        self.leader_steering = 0.0
        self.bridge = CvBridge()
        self.pid_controllers = {i: PIDController(Kp=0.8, Ki=0.01, Kd=0.2) for i in range(3)}
        self.truck_views = {i: None for i in range(3)}
        self.ss_masks_bev = {i: None for i in range(3)}
        
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
        
        # lane-change timing tunables
        self.lc_dt = 0.1
        self.lc_step = 0.05
        
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

        except Exception as e:
            self.get_logger().error(f"Failed to connect to CARLA or find actors: {e}")
            self.world = None

    # -------------------- 라이다 가드 --------------------
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
            # 수동(MANUAL) 명령일 때는 IDLE 상태여도 종료하지 않도록 조건 수정
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

    # -------------------- ROS Callbacks --------------------
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
        steering_angle = calculate_steering(
            self.pid_controllers[truck_id],
            lane_positions,
            frame.shape[1],
            self.current_target_lane[truck_id],
            self.transition_factor[truck_id],
            is_lane_changing=is_changing
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
        
        def update_transition():
            if self.current_target_lane.get(truck_id) != direction:
                if self.transition_timer.get(truck_id):
                    self.transition_timer[truck_id].cancel()
                    self.transition_timer[truck_id] = None
                return

            self.transition_factor[truck_id] += self.lc_step
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

    def start_reorder_maneuver(self, direction):
        if self.maneuver_state != ManeuverState.IDLE:
            self.get_logger().warn("이미 재배치 진행 중")
            return False
        self.reorder_direction = direction
        self.exiting_leader_id = self.truck_order[0]
        self.successor_id = self.truck_order[1]
        self.follower_id = self.truck_order[2]
        self.maneuver_state = ManeuverState.LEADER_EXITS_LANE
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
                self.get_logger().info(
                    f"[REJOIN by Coords] 조건 만족 (전후방: {forward_dist:.1f}m, 좌우: {lateral_dist:.1f}m). "
                    f"리더 {leader_id}가 {reenter_dir} 방향으로 재합류 시작."
                )
                self._guarded_lane_change(leader_id, reenter_dir, tag="REJOIN")
                self._rejoin_timer = None
                return
            else:
                self.get_logger().debug(f"[REJOIN by Coords] 재합류 대기... (전후방: {forward_dist:.1f}m, 좌우: {lateral_dist:.1f}m)")

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
                    ManeuverState.SUCCESSOR_ENTERS_GAP,
                    ManeuverState.FOLLOWER_ENTERS_GAP,
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
                    self.platooning_manager[truck_id].update_distance(
                        lidar_distance, emergency_stop=self.emergency_stop
                    )

def opencv_loop(node: LaneFollowingNode):
    window_name = "Combined Bird-Eye View"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("\n--- OpenCV 창 활성 ---")
    print(">>> 전체 재배치 (선두->후미): 't'(왼쪽), 'y'(오른쪽)")
    print(">>> 후행차량 선두이동: 'j'(2번->1번), 'k'(3번->1번) - 왼쪽으로만 동작")
    print("---------------------------------")
    print(">>> 개별 차선 변경 (기동 중 비활성)")
    print(">>> Truck 0: 'q'(왼), 'e'(오)")
    print(">>> Truck 1: 'a'(왼), 'd'(오)")
    print(">>> Truck 2: 'z'(왼), 'c'(오)")
    print(">>> 'ESC' 종료")
    while rclpy.ok():
        views = []
        for i in node.truck_order:
            v = node.truck_views.get(i)
            if v is None:
                v = np.zeros((480, 640, 3), dtype=np.uint8)
            views.append(v)
        if len(views) == 3:
            cv2.imshow(window_name, cv2.hconcat(views))
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('t'):
            print(">>> 재배치(왼쪽)")
            node.change_queue.put(('reorder', 'left'))
        elif key == ord('y'):
            print(">>> 재배치(오른쪽)")
            node.change_queue.put(('reorder', 'right'))
        elif key == ord('j'):
            if len(node.truck_order) > 1:
                target = node.truck_order[1]
                print(f">>> Promote(왼쪽): Truck {target}을 선두로")
                node.change_queue.put(('promote', target, 'left'))
        elif key == ord('k'):
            if len(node.truck_order) > 2:
                target = node.truck_order[2]
                print(f">>> Promote(왼쪽): Truck {target}을 선두로")
                node.change_queue.put(('promote', target, 'left'))
        elif key == ord('q'): node.change_queue.put((0, 'left'))
        elif key == ord('e'): node.change_queue.put((0, 'right'))
        elif key == ord('a'): node.change_queue.put((1, 'left'))
        elif key == ord('d'): node.change_queue.put((1, 'right'))
        elif key == ord('z'): node.change_queue.put((2, 'left'))
        elif key == ord('c'): node.change_queue.put((2, 'right'))
        elif key == 27:
            print("ESC 입력. 종료")
            break
    cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowingNode()
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()
    print("ROS2 노드 스핀(백그라운드) 시작")
    try:
        opencv_loop(node)
    except KeyboardInterrupt:
        print("Ctrl+C 입력. 종료")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("ROS2 노드/스레드 종료")

if __name__ == '__main__':
    main()
