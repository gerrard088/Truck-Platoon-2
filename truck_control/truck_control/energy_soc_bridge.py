import csv
import fcntl
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, Int32MultiArray
from std_msgs.msg import Int32

try:
    import matlab.engine  # type: ignore
except Exception:  # pragma: no cover
    matlab = None


@dataclass
class TruckState:
    velocity_ms: float = 0.0
    accel_ms2: float = 0.0
    pitch_deg: float = 0.0
    last_z: Optional[float] = None
    pose_x: Optional[float] = None
    pose_y: Optional[float] = None
    pose_z: Optional[float] = None
    grade_deg: float = 0.0
    model_accel_ms2: float = 0.0
    prev_velocity_ms: Optional[float] = None
    soc: float = 100.0
    energy_wh: float = 0.0
    distance_m: float = 0.0
    wh_per_km: float = 0.0
    battery_power_w: float = 0.0
    throttle: float = float("nan")


class EnergySocBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("energy_soc_bridge")

        self.declare_parameter("truck_count", 3)
        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("log_dir", "/home/tmo/ros2_ws/log/energy")
        self.declare_parameter("use_matlab_engine", True)
        self.declare_parameter("matlab_model_dir", "")
        self.declare_parameter("mass_kg", 44000.0)
        self.declare_parameter("rolling_resistance", 0.006)
        self.declare_parameter("cd_a", 5.9)
        self.declare_parameter("air_density", 1.225)
        self.declare_parameter("drive_efficiency", 0.92)
        self.declare_parameter("regen_efficiency", 0.70)
        self.declare_parameter("aux_power_w", 3000.0)
        self.declare_parameter("battery_capacity_kwh", 450.0)
        self.declare_parameter("initial_soc", 100.0)
        self.declare_parameter("platoon_order_topic", "/platoon_order")
        self.declare_parameter("drive_mode_topic", "/drive_mode")
        self.declare_parameter("aero_drag_multipliers", [1.0, 0.75, 0.85])
        self.declare_parameter("use_pose_grade", True)
        self.declare_parameter("grade_smoothing_alpha", 0.25)
        self.declare_parameter("min_grade_distance_m", 0.5)
        self.declare_parameter("max_abs_grade_deg", 8.0)
        self.declare_parameter("use_velocity_diff_accel", True)
        self.declare_parameter("accel_smoothing_alpha", 0.35)
        self.declare_parameter("max_abs_accel_ms2", 6.0)

        self.truck_count = int(self.get_parameter("truck_count").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.log_dir = str(self.get_parameter("log_dir").value)
        self.use_matlab_engine = bool(self.get_parameter("use_matlab_engine").value)
        self.mass_kg = float(self.get_parameter("mass_kg").value)
        self.rolling_resistance = float(self.get_parameter("rolling_resistance").value)
        self.cd_a = float(self.get_parameter("cd_a").value)
        self.air_density = float(self.get_parameter("air_density").value)
        self.drive_efficiency = float(self.get_parameter("drive_efficiency").value)
        self.regen_efficiency = float(self.get_parameter("regen_efficiency").value)
        self.aux_power_w = float(self.get_parameter("aux_power_w").value)
        self.battery_capacity_kwh = float(self.get_parameter("battery_capacity_kwh").value)
        initial_soc = float(self.get_parameter("initial_soc").value)
        self.platoon_order_topic = str(self.get_parameter("platoon_order_topic").value)
        self.drive_mode_topic = str(self.get_parameter("drive_mode_topic").value)
        raw_drag_multipliers = self.get_parameter("aero_drag_multipliers").value
        self.use_pose_grade = bool(self.get_parameter("use_pose_grade").value)
        self.grade_smoothing_alpha = float(self.get_parameter("grade_smoothing_alpha").value)
        self.min_grade_distance_m = float(self.get_parameter("min_grade_distance_m").value)
        self.max_abs_grade_deg = float(self.get_parameter("max_abs_grade_deg").value)
        self.use_velocity_diff_accel = bool(self.get_parameter("use_velocity_diff_accel").value)
        self.accel_smoothing_alpha = float(self.get_parameter("accel_smoothing_alpha").value)
        self.max_abs_accel_ms2 = float(self.get_parameter("max_abs_accel_ms2").value)

        self.states: Dict[int, TruckState] = {i: TruckState(soc=initial_soc) for i in range(self.truck_count)}
        self.truck_rank_by_id: Dict[int, int] = {i: i for i in range(self.truck_count)}
        self.aero_drag_multipliers = self._sanitize_drag_multipliers(raw_drag_multipliers)
        self.last_update_time = self.get_clock().now()
        self.measurements_frozen = False
        self.logging_enabled = False

        self.soc_publishers: Dict[int, Any] = {}
        self.power_publishers: Dict[int, Any] = {}
        self.whkm_publishers: Dict[int, Any] = {}

        self._csv_files: Dict[int, Any] = {}
        self._csv_writers: Dict[int, csv.writer] = {}
        self._log_lock_file = None

        self._matlab_engine = None
        self._matlab_available = False
        self._configure_matlab_engine()
        self._configure_loggers()
        self._configure_io()

        period = 1.0 / max(1.0, self.publish_hz)
        self.timer = self.create_timer(period, self._on_timer)
        self.get_logger().info(
            f"energy_soc_bridge started: trucks={self.truck_count}, hz={self.publish_hz:.1f}, "
            f"matlab_mode={self._matlab_available}, aero_drag_multipliers={self.aero_drag_multipliers}"
        )

    def _sanitize_drag_multipliers(self, values: Any) -> list:
        multipliers = []
        if isinstance(values, (list, tuple)):
            for value in values:
                try:
                    multipliers.append(max(0.05, float(value)))
                except (TypeError, ValueError):
                    continue

        if not multipliers:
            multipliers = [1.0]

        while len(multipliers) < self.truck_count:
            multipliers.append(multipliers[-1])

        return multipliers[:self.truck_count]

    def _configure_matlab_engine(self) -> None:
        if not self.use_matlab_engine or matlab is None:
            self.get_logger().warn("MATLAB engine unavailable or disabled. Using Python fallback model.")
            return

        try:
            self._matlab_engine = matlab.engine.start_matlab()
            model_dir = str(self.get_parameter("matlab_model_dir").value).strip()
            if model_dir:
                self._matlab_engine.addpath(model_dir, nargout=0)
            else:
                share_dir = os.path.join(get_package_share_directory("truck_control"), "matlab")
                source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "matlab"))
                if os.path.isdir(share_dir):
                    self._matlab_engine.addpath(share_dir, nargout=0)
                if os.path.isdir(source_dir):
                    self._matlab_engine.addpath(source_dir, nargout=0)

            # Check that helper exists on MATLAB path.
            exists_flag = self._matlab_engine.exist("eplatoon_power_step", "file")
            if int(exists_flag) == 2:
                self._matlab_available = True
            else:
                self.get_logger().warn("MATLAB function eplatoon_power_step.m not found. Falling back to Python model.")
        except Exception as exc:
            self.get_logger().warn(f"Failed to start MATLAB engine ({exc}). Using Python fallback model.")
            self._matlab_available = False

    def _configure_loggers(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        self._acquire_log_lock()
        for truck_id in range(self.truck_count):
            path = os.path.join(self.log_dir, f"truck{truck_id}_energy.csv")
            f = open(path, "w", newline="", encoding="utf-8", buffering=1)
            w = csv.writer(f)
            w.writerow([
                "time_sec",
                "velocity_ms",
                "accel_ms2",
                "pitch_deg",
                "model_accel_ms2",
                "model_grade_deg",
                "battery_power_w",
                "soc",
                "wh_per_km",
                "throttle",
            ])
            f.flush()
            self._csv_files[truck_id] = f
            self._csv_writers[truck_id] = w

    def _acquire_log_lock(self) -> None:
        lock_path = os.path.join(self.log_dir, ".energy_soc_bridge.lock")
        self._log_lock_file = open(lock_path, "w", encoding="utf-8")
        try:
            fcntl.flock(self._log_lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._log_lock_file.close()
            self._log_lock_file = None
            raise RuntimeError(
                f"log_dir is already locked by another energy_soc_bridge process: {self.log_dir}"
            ) from exc

        self._log_lock_file.seek(0)
        self._log_lock_file.truncate()
        self._log_lock_file.write(f"{os.getpid()}\n")
        self._log_lock_file.flush()

    def _configure_io(self) -> None:
        self.create_subscription(Int32MultiArray, self.platoon_order_topic, self._order_cb, 10)
        self.create_subscription(Int32, self.drive_mode_topic, self._drive_mode_cb, 10)
        for truck_id in range(self.truck_count):
            self.create_subscription(
                Float32, f"/truck{truck_id}/velocity", lambda msg, i=truck_id: self._velocity_cb(msg, i), 10
            )
            self.create_subscription(
                Float32MultiArray, f"/truck{truck_id}/accel", lambda msg, i=truck_id: self._accel_cb(msg, i), 10
            )
            self.create_subscription(
                Float32, f"/truck{truck_id}/pitch_deg", lambda msg, i=truck_id: self._pitch_cb(msg, i), 10
            )
            self.create_subscription(
                PoseStamped, f"/truck{truck_id}/pose3d", lambda msg, i=truck_id: self._pose_cb(msg, i), 10
            )
            self.create_subscription(
                Float32, f"/truck{truck_id}/throttle_control", lambda msg, i=truck_id: self._throttle_cb(msg, i), 10
            )

            self.soc_publishers[truck_id] = self.create_publisher(Float32, f"/truck{truck_id}/battery_soc", 10)
            self.power_publishers[truck_id] = self.create_publisher(Float32, f"/truck{truck_id}/battery_power_w", 10)
            self.whkm_publishers[truck_id] = self.create_publisher(Float32, f"/truck{truck_id}/energy_wh_per_km", 10)

    def _velocity_cb(self, msg: Float32, truck_id: int) -> None:
        self.states[truck_id].velocity_ms = float(msg.data)

    def _accel_cb(self, msg: Float32MultiArray, truck_id: int) -> None:
        if msg.data:
            self.states[truck_id].accel_ms2 = float(msg.data[0])

    def _pitch_cb(self, msg: Float32, truck_id: int) -> None:
        self.states[truck_id].pitch_deg = float(msg.data)

    def _throttle_cb(self, msg: Float32, truck_id: int) -> None:
        self.states[truck_id].throttle = float(msg.data)

    def _pose_cb(self, msg: PoseStamped, truck_id: int) -> None:
        st = self.states[truck_id]
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        z = float(msg.pose.position.z)

        if self.use_pose_grade and st.pose_x is not None and st.pose_y is not None and st.pose_z is not None:
            dx = x - st.pose_x
            dy = y - st.pose_y
            dz = z - st.pose_z
            planar_distance = math.hypot(dx, dy)
            if planar_distance >= self.min_grade_distance_m:
                instant_grade_deg = math.degrees(math.atan2(dz, planar_distance))
                instant_grade_deg = max(-self.max_abs_grade_deg, min(self.max_abs_grade_deg, instant_grade_deg))
                alpha = max(0.0, min(1.0, self.grade_smoothing_alpha))
                st.grade_deg = ((1.0 - alpha) * st.grade_deg) + (alpha * instant_grade_deg)

        st.last_z = z
        st.pose_x = x
        st.pose_y = y
        st.pose_z = z

    def _order_cb(self, msg: Int32MultiArray) -> None:
        order = [int(truck_id) for truck_id in msg.data]
        if len(order) != self.truck_count:
            self.get_logger().warn(f"Ignoring invalid platoon order length: {order}")
            return

        expected_ids = set(range(self.truck_count))
        if set(order) != expected_ids:
            self.get_logger().warn(f"Ignoring invalid platoon order ids: {order}")
            return

        rank_by_id = {truck_id: rank for rank, truck_id in enumerate(order)}
        if rank_by_id == self.truck_rank_by_id:
            return

        self.truck_rank_by_id = rank_by_id
        self.get_logger().info(f"Updated platoon order for energy model: {order}")

    def _drive_mode_cb(self, msg: Int32) -> None:
        mode = int(msg.data)
        was_enabled = self.logging_enabled
        self.logging_enabled = mode > 0
        if self.logging_enabled and not was_enabled:
            self.last_update_time = self.get_clock().now()
            self.get_logger().info(f"Drive mode selected ({mode}). Energy logging enabled.")

    def _on_timer(self) -> None:
        if self.measurements_frozen or not self.logging_enabled:
            return

        now = self.get_clock().now()
        dt = (now - self.last_update_time).nanoseconds / 1e9
        if dt <= 0.0:
            return
        self.last_update_time = now

        stamp_sec = now.nanoseconds / 1e9
        depleted_ids = []
        for truck_id in range(self.truck_count):
            st = self.states[truck_id]
            model_accel_ms2 = self._effective_accel_ms2(st, dt)
            model_grade_deg = self._effective_grade_deg(st)
            power_w = self._estimate_power_w(truck_id, st, model_accel_ms2, model_grade_deg)
            st.battery_power_w = power_w

            delta_wh = power_w * (dt / 3600.0)
            st.energy_wh += delta_wh
            st.distance_m += max(0.0, st.velocity_ms) * dt

            if self.battery_capacity_kwh > 0.0:
                delta_soc = (delta_wh / 1000.0) / self.battery_capacity_kwh * 100.0
                st.soc = max(0.0, min(100.0, st.soc - delta_soc))
                if st.soc <= 0.0:
                    depleted_ids.append(truck_id)

            if st.distance_m > 1.0:
                st.wh_per_km = st.energy_wh / (st.distance_m / 1000.0)
            else:
                st.wh_per_km = 0.0

            self._publish_one(truck_id, st)
            self._csv_writers[truck_id].writerow(
                [f"{stamp_sec:.3f}", f"{st.velocity_ms:.4f}", f"{st.accel_ms2:.4f}", f"{st.pitch_deg:.4f}",
                 f"{st.model_accel_ms2:.4f}", f"{model_grade_deg:.4f}", f"{st.battery_power_w:.2f}",
                 f"{st.soc:.4f}", f"{st.wh_per_km:.4f}", f"{st.throttle:.4f}"]
            )
            self._csv_files[truck_id].flush()

        if depleted_ids:
            self.measurements_frozen = True
            self.get_logger().warn(
                f"SOC depleted for truck(s) {depleted_ids}. Freezing all energy measurements."
            )

    def _effective_accel_ms2(self, st: TruckState, dt: float) -> float:
        if not self.use_velocity_diff_accel or dt <= 0.0:
            st.model_accel_ms2 = st.accel_ms2
            st.prev_velocity_ms = st.velocity_ms
            return st.model_accel_ms2

        if st.prev_velocity_ms is None:
            st.prev_velocity_ms = st.velocity_ms
            st.model_accel_ms2 = 0.0
            return st.model_accel_ms2

        raw_accel = (st.velocity_ms - st.prev_velocity_ms) / dt
        st.prev_velocity_ms = st.velocity_ms
        raw_accel = max(-self.max_abs_accel_ms2, min(self.max_abs_accel_ms2, raw_accel))
        alpha = max(0.0, min(1.0, self.accel_smoothing_alpha))
        st.model_accel_ms2 = ((1.0 - alpha) * st.model_accel_ms2) + (alpha * raw_accel)
        return st.model_accel_ms2

    def _effective_grade_deg(self, st: TruckState) -> float:
        if self.use_pose_grade and st.pose_x is not None and st.pose_y is not None and st.pose_z is not None:
            return st.grade_deg
        return st.pitch_deg

    def _effective_cd_a(self, truck_id: int) -> float:
        rank = self.truck_rank_by_id.get(truck_id, truck_id)
        multiplier = self.aero_drag_multipliers[min(rank, len(self.aero_drag_multipliers) - 1)]
        return self.cd_a * multiplier

    def _estimate_power_w(self, truck_id: int, st: TruckState, accel_ms2: float, grade_deg: float) -> float:
        v = max(0.0, st.velocity_ms)
        a = accel_ms2
        pitch_deg = grade_deg
        effective_cd_a = self._effective_cd_a(truck_id)

        if self._matlab_available and self._matlab_engine is not None:
            try:
                result = self._matlab_engine.eplatoon_power_step(
                    float(v),
                    float(a),
                    float(pitch_deg),
                    float(self.mass_kg),
                    float(self.rolling_resistance),
                    float(effective_cd_a),
                    float(self.air_density),
                    float(self.drive_efficiency),
                    float(self.regen_efficiency),
                    float(self.aux_power_w),
                    nargout=1,
                )
                return float(result)
            except Exception as exc:
                self.get_logger().warn(f"MATLAB call failed, switching to Python model: {exc}")
                self._matlab_available = False

        return self._python_power_step(v, a, pitch_deg, effective_cd_a)

    def _python_power_step(self, v: float, a: float, pitch_deg: float, cd_a: float) -> float:
        g = 9.81
        pitch_rad = math.radians(pitch_deg)
        grade_force = self.mass_kg * g * math.sin(pitch_rad)
        rolling_force = self.mass_kg * g * self.rolling_resistance
        aero_force = 0.5 * self.air_density * cd_a * (v ** 2)
        inertial_force = self.mass_kg * a
        total_force = grade_force + rolling_force + aero_force + inertial_force
        wheel_power = total_force * v

        if wheel_power >= 0.0:
            batt_power = wheel_power / max(0.05, self.drive_efficiency) + self.aux_power_w
        else:
            # Regen disabled: decel energy is dissipated mechanically, battery only supplies auxiliaries.
            batt_power = self.aux_power_w
        return batt_power

    def _publish_one(self, truck_id: int, st: TruckState) -> None:
        soc_msg = Float32()
        soc_msg.data = float(st.soc)
        self.soc_publishers[truck_id].publish(soc_msg)

        power_msg = Float32()
        power_msg.data = float(st.battery_power_w)
        self.power_publishers[truck_id].publish(power_msg)

        whkm_msg = Float32()
        whkm_msg.data = float(st.wh_per_km)
        self.whkm_publishers[truck_id].publish(whkm_msg)

    def destroy_node(self) -> bool:
        for f in self._csv_files.values():
            try:
                f.close()
            except Exception:
                pass
        if self._log_lock_file is not None:
            try:
                fcntl.flock(self._log_lock_file.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                self._log_lock_file.close()
            except Exception:
                pass
            self._log_lock_file = None
        if self._matlab_engine is not None:
            try:
                self._matlab_engine.quit()
            except Exception:
                pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EnergySocBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
