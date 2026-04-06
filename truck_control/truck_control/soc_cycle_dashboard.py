import csv
import os
import signal
import time
import tkinter as tk
from dataclasses import dataclass
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32, Int32MultiArray


@dataclass
class TruckCycleState:
    speed_ms: float = 0.0
    soc: float = 0.0
    power_w: float = 0.0
    wh_per_km: float = 0.0
    received_soc: bool = False


class SocCycleSubscriber(Node):
    def __init__(self, dashboard, truck_count: int):
        super().__init__("soc_cycle_dashboard")
        self.dashboard = dashboard
        self.truck_count = truck_count
        self.states: Dict[int, TruckCycleState] = {i: TruckCycleState() for i in range(self.truck_count)}
        self.measurements_frozen = False
        self.current_maneuver_count = 0
        self.last_recorded_maneuver_count = 0
        self.maneuver_elapsed_sec = 0.0
        self.last_maneuver_duration_sec = 0.0
        self.truck_roles: Dict[int, str] = {0: "Leader", 1: "Mid", 2: "Tail"}
        self.total_drive_time_sec = 0.0
        self.total_drive_distance_m = 0.0
        self._last_update_monotonic = time.monotonic()
        self.log_dir = "/home/tmo/ros2_ws/log/energy"
        self.log_path = os.path.join(self.log_dir, "per_lap_energy_summary.txt")
        self.csv_log_path = os.path.join(self.log_dir, "per_lap_energy_summary.csv")
        self._csv_log_file = None
        self._csv_writer = None

        os.makedirs(self.log_dir, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("Per-lap energy summary\n")
            f.write("======================\n\n")
        self._init_csv_log()

        self.create_subscription(Float32, "/platoon_maneuver_elapsed_sec", self._maneuver_elapsed_cb, 10)
        self.create_subscription(Float32, "/platoon_maneuver_last_duration_sec", self._maneuver_last_duration_cb, 10)
        self.create_subscription(Int32, "/platoon_maneuver_count", self._maneuver_count_cb, 10)
        self.create_subscription(Int32MultiArray, "/platoon_order", self._platoon_order_cb, 10)

        for truck_id in range(self.truck_count):
            self.create_subscription(
                Float32, f"/truck{truck_id}/velocity", lambda msg, i=truck_id: self._speed_cb(msg, i), 10
            )
            self.create_subscription(
                Float32, f"/truck{truck_id}/battery_soc", lambda msg, i=truck_id: self._soc_cb(msg, i), 10
            )
            self.create_subscription(
                Float32, f"/truck{truck_id}/battery_power_w", lambda msg, i=truck_id: self._power_cb(msg, i), 10
            )
            self.create_subscription(
                Float32, f"/truck{truck_id}/energy_wh_per_km", lambda msg, i=truck_id: self._whkm_cb(msg, i), 10
            )

        self.create_timer(0.1, self._update_ui)

    def _speed_cb(self, msg: Float32, truck_id: int) -> None:
        if self.measurements_frozen:
            return
        self.states[truck_id].speed_ms = max(0.0, float(msg.data))

    def _soc_cb(self, msg: Float32, truck_id: int) -> None:
        if self.measurements_frozen:
            return

        state = self.states[truck_id]
        state.soc = max(0.0, min(100.0, float(msg.data)))
        state.received_soc = True

        if state.soc <= 0.0:
            self.measurements_frozen = True
            self.get_logger().warn(
                f"Truck {truck_id} SOC depleted. Freezing SoC cycle history."
            )

    def _power_cb(self, msg: Float32, truck_id: int) -> None:
        if self.measurements_frozen:
            return
        self.states[truck_id].power_w = float(msg.data)

    def _whkm_cb(self, msg: Float32, truck_id: int) -> None:
        if self.measurements_frozen:
            return
        self.states[truck_id].wh_per_km = float(msg.data)

    def _maneuver_elapsed_cb(self, msg: Float32) -> None:
        self.maneuver_elapsed_sec = float(msg.data)

    def _maneuver_last_duration_cb(self, msg: Float32) -> None:
        self.last_maneuver_duration_sec = float(msg.data)

    def _maneuver_count_cb(self, msg: Int32) -> None:
        self.current_maneuver_count = max(0, int(msg.data))

    def _platoon_order_cb(self, msg: Int32MultiArray) -> None:
        order = [int(truck_id) for truck_id in msg.data]
        if len(order) != self.truck_count:
            return

        roles = ["Leader", "Mid", "Tail"]
        for rank, truck_id in enumerate(order):
            self.truck_roles[truck_id] = roles[min(rank, len(roles) - 1)]

    def _update_drive_totals(self) -> None:
        now = time.monotonic()
        dt = max(0.0, now - self._last_update_monotonic)
        self._last_update_monotonic = now

        if (not self.measurements_frozen) and any(state.speed_ms > 0.1 for state in self.states.values()):
            self.total_drive_time_sec += dt
            self.total_drive_distance_m += sum(max(0.0, state.speed_ms) * dt for state in self.states.values())

    def _update_ui(self) -> None:
        self._update_drive_totals()

        if self.measurements_frozen:
            self.dashboard.set_status("Measurements frozen because SOC reached 0%.")
            return

        if not all(state.received_soc for state in self.states.values()):
            self.dashboard.set_status("Waiting for SOC topics...")
            return

        if self.current_maneuver_count <= 0:
            self.dashboard.set_status("Waiting for first maneuver...")
            return

        if self.current_maneuver_count == self.last_recorded_maneuver_count:
            self.dashboard.set_status(
                f"Waiting for next maneuver... current count: {self.current_maneuver_count}"
            )
            return

        while self.last_recorded_maneuver_count < self.current_maneuver_count:
            self.last_recorded_maneuver_count += 1
            cycle_slot = self.last_recorded_maneuver_count - 1
            cycle_socs = [self.states[truck_id].soc for truck_id in range(self.truck_count)]
            cycle_avg_soc = sum(cycle_socs) / max(1, self.truck_count)
            self.dashboard.append_cycle(cycle_slot, cycle_socs, cycle_avg_soc)
            self._append_cycle_log(cycle_slot, cycle_socs, cycle_avg_soc)

    def _append_cycle_log(self, cycle: int, cycle_socs: List[float], avg_soc: float) -> None:
        fleet_efficiency = sum(self.states[truck_id].wh_per_km for truck_id in range(self.truck_count)) / max(1, self.truck_count)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"Lap {cycle}\n")
            f.write(f"  maneuver_count: {self.current_maneuver_count}\n")
            f.write(f"  fleet_avg_soc_pct: {avg_soc:.3f}\n")
            f.write(f"  fleet_sum_soc_pct: {sum(cycle_socs):.3f}\n")
            f.write(f"  fleet_efficiency_wh_per_km: {fleet_efficiency:.3f}\n")
            f.write(f"  total_drive_time_hms: {self._format_hms(self.total_drive_time_sec)}\n")
            f.write(f"  total_drive_distance_km: {self.total_drive_distance_m / 1000.0:.3f}\n")
            f.write(f"  maneuver_elapsed_sec: {self.maneuver_elapsed_sec:.3f}\n")
            f.write(f"  last_maneuver_duration_sec: {self.last_maneuver_duration_sec:.3f}\n")
            for truck_id in range(self.truck_count):
                state = self.states[truck_id]
                f.write(
                    f"  truck{truck_id}: role={self.truck_roles.get(truck_id, 'Unknown')}, "
                    f"speed_kmh={state.speed_ms * 3.6:.3f}, soc_pct={state.soc:.3f}, "
                    f"power_kw={state.power_w / 1000.0:.3f}, efficiency_wh_per_km={state.wh_per_km:.3f}\n"
                )
            f.write("\n")
        self._append_cycle_csv(cycle, cycle_socs, avg_soc, fleet_efficiency)

    def _init_csv_log(self) -> None:
        self._csv_log_file = open(self.csv_log_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_log_file)
        header = [
            "lap",
            "maneuver_count",
            "fleet_avg_soc_pct",
            "fleet_sum_soc_pct",
            "fleet_efficiency_wh_per_km",
            "total_drive_time_sec",
            "total_drive_distance_km",
            "maneuver_elapsed_sec",
            "last_maneuver_duration_sec",
        ]
        for truck_id in range(self.truck_count):
            header.extend([
                f"truck{truck_id}_role",
                f"truck{truck_id}_speed_kmh",
                f"truck{truck_id}_soc_pct",
                f"truck{truck_id}_power_kw",
                f"truck{truck_id}_efficiency_wh_per_km",
            ])
        self._csv_writer.writerow(header)
        self._csv_log_file.flush()

    def _append_cycle_csv(self, cycle: int, cycle_socs: List[float], avg_soc: float, fleet_efficiency: float) -> None:
        if self._csv_writer is None or self._csv_log_file is None:
            return

        row = [
            cycle,
            self.current_maneuver_count,
            round(avg_soc, 3),
            round(sum(cycle_socs), 3),
            round(fleet_efficiency, 3),
            round(self.total_drive_time_sec, 3),
            round(self.total_drive_distance_m / 1000.0, 3),
            round(self.maneuver_elapsed_sec, 3),
            round(self.last_maneuver_duration_sec, 3),
        ]
        for truck_id in range(self.truck_count):
            state = self.states[truck_id]
            row.extend([
                self.truck_roles.get(truck_id, "Unknown"),
                round(state.speed_ms * 3.6, 3),
                round(state.soc, 3),
                round(state.power_w / 1000.0, 3),
                round(state.wh_per_km, 3),
            ])

        self._csv_writer.writerow(row)
        self._csv_log_file.flush()

    @staticmethod
    def _format_hms(total_seconds: float) -> str:
        total_seconds = max(0, int(total_seconds))
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class SocCycleDashboardUI:
    def __init__(self, root: tk.Tk, truck_count: int = 3):
        self.root = root
        self.root.title("Truck SOC Cycle Dashboard")
        self.root.configure(bg="#F3F6FA")
        self.truck_count = truck_count

        self.cycles: List[int] = []
        self.avg_soc_history: List[float] = []
        self.soc_history_by_truck: Dict[int, List[float]] = {i: [] for i in range(self.truck_count)}
        self.bar_offsets = [-0.26, 0.0, 0.26]
        self.bar_width = 0.24
        self.truck_colors = ["#1F77B4", "#FF7F0E", "#2CA02C"]

        self._build_layout()

    def _build_layout(self) -> None:
        header = tk.Frame(self.root, bg="#F3F6FA")
        header.pack(fill=tk.X, padx=14, pady=(10, 4))

        title = tk.Label(
            header,
            text="Cycle-by-Cycle SOC History",
            font=("Helvetica", 18, "bold"),
            bg="#F3F6FA",
            fg="#132238",
        )
        title.pack(side=tk.LEFT)

        self.status_label = tk.Label(
            header,
            text="Waiting for SOC topics...",
            font=("Helvetica", 11),
            bg="#F3F6FA",
            fg="#4A5568",
        )
        self.status_label.pack(side=tk.RIGHT)

        self.figure, self.ax = plt.subplots(figsize=(13, 6))
        self.figure.patch.set_facecolor("#F3F6FA")
        self.ax.set_facecolor("white")

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        self._redraw()

    def set_status(self, text: str) -> None:
        self.status_label.config(text=text)

    def append_cycle(self, cycle: int, cycle_socs: List[float], avg_soc: float) -> None:
        self.cycles.append(cycle)
        self.avg_soc_history.append(avg_soc)
        for truck_id, soc in enumerate(cycle_socs):
            self.soc_history_by_truck[truck_id].append(soc)

        self.status_label.config(
            text=f"Completed Laps: {cycle + 1} | Fleet Avg SOC: {avg_soc:.1f} %"
        )
        self._redraw()

    def _redraw(self) -> None:
        self.ax.clear()
        self.ax.set_facecolor("white")

        if not self.cycles:
            self.ax.set_title("SOC History will appear after SOC topics arrive")
            self.ax.set_xlabel("Lap")
            self.ax.set_ylabel("SOC (%)")
            self.ax.set_ylim(30.0, 100.0)
            self.ax.grid(True, axis="y", linestyle="--", alpha=0.3)
            self.canvas.draw()
            return

        for truck_id in range(self.truck_count):
            x_positions = [cycle + self.bar_offsets[truck_id] for cycle in self.cycles]
            self.ax.bar(
                x_positions,
                self.soc_history_by_truck[truck_id],
                width=self.bar_width,
                color=self.truck_colors[truck_id % len(self.truck_colors)],
                alpha=0.78,
                label=f"Truck {truck_id} SOC",
                zorder=2,
            )

        self.ax.plot(
            self.cycles,
            self.avg_soc_history,
            color="#C0392B",
            linewidth=2.4,
            marker="o",
            markersize=4.0,
            label="Fleet Avg SOC",
            zorder=3,
        )

        x_min = self.cycles[0]
        x_max = self.cycles[-1]
        tick_step = max(1, len(self.cycles) // 12)
        self.ax.set_xticks(list(range(int(x_min), int(x_max) + 1, tick_step)))
        self.ax.set_xlim(x_min - 0.8, x_max + 0.8)
        self.ax.set_ylim(30.0, 100.0)
        self.ax.set_title("Per-Lap Truck SOC Bars with Fleet Average SOC Line", pad=12)
        self.ax.set_xlabel("Lap")
        self.ax.set_ylabel("SOC (%)")
        self.ax.grid(True, axis="y", linestyle="--", alpha=0.3, zorder=1)
        self.ax.legend(loc="upper right")
        self.figure.tight_layout()
        self.canvas.draw()


def main(args=None) -> None:
    rclpy.init(args=args)
    root = tk.Tk()
    ui = SocCycleDashboardUI(root, truck_count=3)
    node = SocCycleSubscriber(ui, truck_count=3)

    def shutdown() -> None:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", shutdown)
    signal.signal(signal.SIGINT, lambda _sig, _frame: shutdown())

    def spin_ros() -> None:
        if rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            root.after(30, spin_ros)

    root.after(30, spin_ros)
    root.mainloop()


if __name__ == "__main__":
    main()
