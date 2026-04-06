import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import time
import math

class PlatooningManager:
    def __init__(self, node: Node, namespace: str):
        """
        플래투닝 전략 모듈 초기화

        이 모듈은 후행 차량이 선두 차량과 동일한 간격(target_distance)을 유지하도록
        V2V 정보(선두 차량의 속도와 가속도)와 LiDAR를 통한 거리 및 경사(pitch) 정보를 기반으로 
        PID 제어를 수행합니다.
        """
        self.node = node
        self.namespace = namespace

        # throttle_control 토픽에 Float32 타입 메시지로 스로틀(속도) 명령을 퍼블리시
        self.throttle_pub = self.node.create_publisher(Float32, f'/{namespace}/throttle_control', 10)

        # LiDAR에서 측정한 정보: 거리와 pitch (pitch는 도 단위)
        self.lidar_distance = None
        self.lidar_pitch = 0.0

        # PID 제어 파라미터
        self.min_distance = 2.0      # 정지 시 최소 간격 (m)
        self.time_gap = 0.6          # 시간 간격 (s)
        self.target_distance = 8.0  # 목표 차량 간 거리 (m)
        self.safe_distance = 2.5     # 최소 안전 간격 (m) 이하이면 저속 혹은 정지
        self.catchup_distance_margin = 1.5  # 목표 거리보다 더 벌어졌을 때 가속 보너스 시작
        self.catchup_speed_gain = 0.2       # 거리 초과분(m)당 추가 목표 속도(m/s)
        self.catchup_speed_cap = 2.0        # 선행차 대비 추가 목표 속도 상한(m/s)
        self.predecessor_speed_weight = 0.65
        self.platoon_leader_speed_weight = 0.35
        self.sensor_loss_speed_gain = 0.25
        self.sensor_loss_command_limit = 0.25
        self.command_alpha = 1.0
        self.last_speed_command = 0.0
        self.kp = 0.3
        self.ki = 0.01
        self.kd = 1.8
        self.integral = 0.0
        self.prev_error = 0.0

        if namespace.endswith('truck2'):
            self.catchup_speed_gain = 0.12
            self.catchup_speed_cap = 1.2
            self.predecessor_speed_weight = 0.4
            self.platoon_leader_speed_weight = 0.6
            self.sensor_loss_speed_gain = 0.18
            self.sensor_loss_command_limit = 0.18
            self.command_alpha = 0.35

        # dt 계산을 위한 이전 시간 초기화
        self.prev_time = time.time()

    def update_lidar_info(self, lidar_msg):
        """
        LiDAR 데이터 메시지를 업데이트합니다.
        lidar_msg는 다음과 같은 속성을 포함해야 합니다:
          - distance: 선두 차량과의 간격 (m)
          - pitch: LiDAR의 pitch 값 (도 단위)
        """
        self.lidar_distance = lidar_msg.distance
        self.lidar_pitch = lidar_msg.pitch
        self.control_speed()

    def update_distance(self, lidar_distance, leader_velocity, emergency_stop=False, ego_velocity=None, platoon_leader_velocity=None):
        self.lidar_distance = lidar_distance
        # 목표 거리 설정 (항상 8m 유지)
        self.target_distance = 8.0

        self.control_speed(
            leader_velocity,
            emergency_stop=emergency_stop,
            ego_velocity=ego_velocity,
            platoon_leader_velocity=platoon_leader_velocity,
        )

    def control_speed(self, leader_velocity, emergency_stop=False, ego_velocity=None, platoon_leader_velocity=None):
        if emergency_stop:
            self.node.get_logger().info(f"[{self.namespace}] 비상 정지")
            throttle_msg = Float32()
            throttle_msg.data = -1.0
            self.throttle_pub.publish(throttle_msg)
            return

        speed_reference = float(leader_velocity)
        if platoon_leader_velocity is not None:
            speed_reference = (
                self.predecessor_speed_weight * float(leader_velocity)
                + self.platoon_leader_speed_weight * float(platoon_leader_velocity)
            )

        if self.lidar_distance is None:
            if ego_velocity is None:
                return
            vel_error = speed_reference - ego_velocity
            raw_speed_command = max(
                -self.sensor_loss_command_limit,
                min(self.sensor_loss_command_limit, vel_error * self.sensor_loss_speed_gain)
            )
            speed_command = (
                self.command_alpha * raw_speed_command
                + (1.0 - self.command_alpha) * self.last_speed_command
            )
            throttle_msg = Float32()
            throttle_msg.data = float(speed_command)
            self.throttle_pub.publish(throttle_msg)
            self.last_speed_command = float(speed_command)
            return

        current_time = time.time()
        dt = current_time - self.prev_time if self.prev_time != 0 else 0.05
        if dt <= 0: dt = 0.05

        # 1. 거리 오차 기반 속도 보정치 계산 (PID)
        distance_error = self.lidar_distance - self.target_distance
        self.integral += distance_error * dt
        derivative = (distance_error - self.prev_error) / dt
        
        # 거리 유지를 위한 보정 속도 (m/s)
        # Kp=0.5: 1m 차이당 0.5m/s(1.8km/h) 가감속
        dist_correction = (0.5 * distance_error +
                           0.01 * self.integral +
                           0.1 * derivative)

        catchup_boost = 0.0
        if distance_error > self.catchup_distance_margin:
            catchup_boost = min(
                self.catchup_speed_cap,
                (distance_error - self.catchup_distance_margin) * self.catchup_speed_gain,
            )

        # 2. 최종 목표 속도 = 선행 차량 속도 + 거리 보정 속도 + 추종 보너스
        target_velocity = speed_reference + dist_correction + catchup_boost
        
        # 3. 목표 속도 추종을 위한 스로틀 계산 (단순 P 제어)
        if ego_velocity is not None:
            vel_error = target_velocity - ego_velocity
            raw_speed_command = vel_error * 0.5
        else:
            # ego_velocity 없을 경우 기존 방식 fallback (단, 베이스가 선행차 속도이므로 더 안정적)
            raw_speed_command = dist_correction * 0.2

        # 안전 거리 미만 시 강제 제동
        if self.lidar_distance < self.safe_distance:
            raw_speed_command = -1.0

        raw_speed_command = max(-1.0, min(1.0, raw_speed_command))
        speed_command = (
            self.command_alpha * raw_speed_command
            + (1.0 - self.command_alpha) * self.last_speed_command
        )

        throttle_msg = Float32()
        throttle_msg.data = float(speed_command)
        self.throttle_pub.publish(throttle_msg)

        self.last_speed_command = float(speed_command)
        self.prev_error = distance_error
        self.prev_time = current_time
