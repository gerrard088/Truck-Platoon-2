import math

import carla
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray


class CarlaSpectatorFollower(Node):
    def __init__(self):
        super().__init__('carla_spectator_follower')
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.target_vehicle = None
        self.target_truck_id = 1
        self.previous_location = None
        self.previous_rotation = None

        self.create_subscription(Int32MultiArray, '/platoon_order', self.order_callback, 10)
        self.update_target_vehicle(force_log=True)

    def order_callback(self, msg: Int32MultiArray):
        if len(msg.data) < 2:
            self.get_logger().warn('platoon_order 메시지 길이가 2보다 작아 mid 차량을 결정할 수 없습니다.')
            return

        mid_truck_id = int(msg.data[1])
        if mid_truck_id != self.target_truck_id:
            self.target_truck_id = mid_truck_id
            self.update_target_vehicle(force_log=True)

    def update_target_vehicle(self, force_log=False):
        target_role = f'truck{self.target_truck_id}'
        actors = self.world.get_actors()
        vehicles = [actor for actor in actors if actor.type_id.startswith('vehicle.daf')]
        matched = [v for v in vehicles if v.attributes.get('role_name') == target_role]

        if matched:
            vehicle = matched[0]
            if self.target_vehicle is None or self.target_vehicle.id != vehicle.id or force_log:
                self.get_logger().info(
                    f"Spectator target -> mid role '{target_role}' (actor id={vehicle.id}, type={vehicle.type_id})"
                )
            self.target_vehicle = vehicle
            return True

        if force_log:
            self.get_logger().warn(f"'{target_role}' 역할 차량을 찾지 못했습니다.")
        self.target_vehicle = None
        return False

    def lerp(self, start, end, alpha):
        return start + (end - start) * alpha

    def lerp_angle(self, start, end, alpha):
        diff = ((end - start + 180) % 360) - 180
        return (start + diff * alpha) % 360

    def get_relative_location(self, vehicle_transform, dx, dy, dz):
        yaw = math.radians(vehicle_transform.rotation.yaw)
        x = vehicle_transform.location.x + dx * math.cos(yaw) - dy * math.sin(yaw)
        y = vehicle_transform.location.y + dx * math.sin(yaw) + dy * math.cos(yaw)
        z = vehicle_transform.location.z + dz
        return carla.Location(x=x, y=y, z=z)

    def follow_vehicle(self):
        spectator = self.world.get_spectator()
        alpha_position = 0.2
        alpha_rotation = 0.03

        try:
            self.get_logger().info('Spectator 시점을 현재 mid 차량에 고정합니다.')

            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.0)
                self.world.wait_for_tick()

                if self.target_vehicle is None:
                    self.update_target_vehicle(force_log=False)
                    continue

                try:
                    vehicle_transform = self.target_vehicle.get_transform()
                except RuntimeError:
                    self.get_logger().warn('현재 target actor가 유효하지 않아 다시 탐색합니다.')
                    self.target_vehicle = None
                    self.previous_location = None
                    self.previous_rotation = None
                    self.update_target_vehicle(force_log=True)
                    continue

                target_location = self.get_relative_location(vehicle_transform, dx=0, dy=0, dz=80)
                target_yaw = vehicle_transform.rotation.yaw + 90

                if self.previous_location is None or self.previous_rotation is None:
                    smooth_x = target_location.x
                    smooth_y = target_location.y
                    smooth_z = target_location.z
                    smooth_yaw = target_yaw
                    self.previous_rotation = carla.Rotation(pitch=-85, yaw=smooth_yaw, roll=0)
                else:
                    smooth_x = self.lerp(self.previous_location.x, target_location.x, alpha_position)
                    smooth_y = self.lerp(self.previous_location.y, target_location.y, alpha_position)
                    smooth_z = self.lerp(self.previous_location.z, target_location.z, alpha_position)
                    smooth_yaw = self.lerp_angle(self.previous_rotation.yaw, target_yaw, alpha_rotation)
                    self.previous_rotation.yaw = smooth_yaw

                spectator_transform = carla.Transform(
                    carla.Location(x=smooth_x, y=smooth_y, z=smooth_z),
                    carla.Rotation(pitch=-85, yaw=smooth_yaw, roll=0)
                )
                spectator.set_transform(spectator_transform)
                self.previous_location = carla.Location(x=smooth_x, y=smooth_y, z=smooth_z)

        except KeyboardInterrupt:
            self.get_logger().info('시뮬레이션 중지.')
        finally:
            self.get_logger().info('프로그램 종료.')


def main(args=None):
    rclpy.init(args=args)
    follower = CarlaSpectatorFollower()
    try:
        follower.follow_vehicle()
    finally:
        follower.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
