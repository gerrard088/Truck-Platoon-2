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
        self.min_distance = 5.0      # 정지 시 최소 간격 (m)
        self.time_gap = 0.8          # 시간 간격 (s)
        self.target_distance = 5.0  # 동적 계산 전 기본값 (m)
        self.safe_distance = 4.5     # 최소 안전 간격 (m) 이하이면 저속 혹은 정지
        self.kp = 0.3
        self.ki = 0.01
        self.kd = 1.8
        self.integral = 0.0
        self.prev_error = 0.0

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

    def update_distance(self, lidar_distance, emergency_stop=False, ego_velocity=None,
                        min_distance_override=None, time_gap_override=None):
        self.lidar_distance = lidar_distance
        if ego_velocity is not None:
            min_dist = self.min_distance if min_distance_override is None else float(min_distance_override)
            time_gap = self.time_gap if time_gap_override is None else float(time_gap_override)
            self.target_distance = max(min_dist, min_dist + time_gap * float(ego_velocity))
        self.control_speed(emergency_stop=emergency_stop)  # ✅ 전달값 그대로 사용

    def control_speed(self, emergency_stop=False):

        if emergency_stop:
            self.node.get_logger().info(f"[{self.namespace}] 비상 정지")
            throttle_msg = Float32()
            throttle_msg.data = -1.0
            self.throttle_pub.publish(throttle_msg)
            return
        """
        V2V와 LiDAR(거리, pitch) 데이터를 결합하여 PID 제어를 수행하고,
        목표 간격을 유지하기 위한 스로틀 명령을 계산하여 퍼블리시합니다.
        """
        if self.lidar_distance is None:
            self.node.get_logger().warn(f"[{self.namespace}] LiDAR 거리 정보 없음, 제어 수행 불가")
            return

        current_time = time.time()
        dt = current_time - self.prev_time if self.prev_time != 0 else 1.0

        # 거리 오차 계산: 측정된 간격과 목표 간격의 차이
        distance_error = self.lidar_distance - self.target_distance

        # PID 제어: 적분 및 미분 값 계산
        self.integral += distance_error * dt
        derivative = (distance_error - self.prev_error) / dt if dt > 0 else 0.0
        pid_correction = (self.kp * distance_error +
                          self.ki * self.integral +
                          self.kd * derivative)

        # 최종 속도 명령 계산
        speed_command =  pid_correction

        # 안전 조건: LiDAR 측정 거리가 안전 간격보다 작으면 정지 혹은 저속 명령
        if self.lidar_distance < self.safe_distance:
            speed_command = -1.0
        # 스로틀 제한
        speed_command = max(-1.0, min(1.0, speed_command))

        # 스로틀 명령 퍼블리시
        throttle_msg = Float32()
        throttle_msg.data = speed_command
        self.throttle_pub.publish(throttle_msg)

        self.prev_error = distance_error
        self.prev_time = current_time
