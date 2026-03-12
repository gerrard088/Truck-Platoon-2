from std_msgs.msg import Float32

def publish_commands(throttle_publishers, current_velocities, target_velocity, last_steering, emergency_stop=False):
    for truck_id in range(len(throttle_publishers)):
    # Truck 0에 대해 emergency_stop인 경우, 브레이크(-1.0) 명령
        if truck_id == 0 and emergency_stop:
            msg = Float32()
            msg.data = -1.0  # throttle 0
            throttle_publishers[truck_id].publish(msg)
            continue

        # 기본적 publish_commands 로직
        throttle_value = 0.8
        if current_velocities[truck_id] > target_velocity:
            throttle_value = 0.0

        steering_abs = abs(last_steering[truck_id])
        if steering_abs > 2.5:
            # 고속 곡선에서 동시 언더/오버슈트를 막기 위해 조향각에 비례해 추가 감속
            if steering_abs >= 10.0:
                steering_scale = 0.2
            else:
                reduction_ratio = (steering_abs - 2.5) / 7.5
                steering_scale = 1.0 - 0.8 * reduction_ratio
            throttle_value *= max(0.2, steering_scale)

        msg = Float32()
        msg.data = throttle_value
        throttle_publishers[truck_id].publish(msg)

if __name__ == '__main__':
    pass
