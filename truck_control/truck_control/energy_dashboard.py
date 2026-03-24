import signal
import tkinter as tk
from dataclasses import dataclass
from typing import Dict

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32, Int32MultiArray


@dataclass
class TruckEnergyState:
    speed_ms: float = 0.0
    soc: float = 0.0
    power_w: float = 0.0
    wh_per_km: float = 0.0


class EnergySubscriber(Node):
    def __init__(self, dashboard, truck_count: int):
        super().__init__("energy_dashboard_subscriber")
        self.dashboard = dashboard
        self.truck_count = truck_count
        self.states: Dict[int, TruckEnergyState] = {i: TruckEnergyState() for i in range(self.truck_count)}
        self.maneuver_elapsed_sec = 0.0
        self.last_maneuver_duration_sec = 0.0
        self.maneuver_count = 0
        self.truck_roles: Dict[int, str] = {0: 'Leader', 1: 'Mid', 2: 'Tail'}

        self.create_subscription(Float32, '/platoon_maneuver_elapsed_sec', self._maneuver_elapsed_cb, 10)
        self.create_subscription(Float32, '/platoon_maneuver_last_duration_sec', self._maneuver_last_duration_cb, 10)
        self.create_subscription(Int32, '/platoon_maneuver_count', self._maneuver_count_cb, 10)
        self.create_subscription(Int32MultiArray, '/platoon_order', self._platoon_order_cb, 10)

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
        self.states[truck_id].speed_ms = float(msg.data)

    def _soc_cb(self, msg: Float32, truck_id: int) -> None:
        self.states[truck_id].soc = float(msg.data)

    def _power_cb(self, msg: Float32, truck_id: int) -> None:
        self.states[truck_id].power_w = float(msg.data)

    def _whkm_cb(self, msg: Float32, truck_id: int) -> None:
        self.states[truck_id].wh_per_km = float(msg.data)

    def _maneuver_elapsed_cb(self, msg: Float32) -> None:
        self.maneuver_elapsed_sec = float(msg.data)

    def _maneuver_last_duration_cb(self, msg: Float32) -> None:
        self.last_maneuver_duration_sec = float(msg.data)

    def _maneuver_count_cb(self, msg: Int32) -> None:
        self.maneuver_count = int(msg.data)

    def _platoon_order_cb(self, msg: Int32MultiArray) -> None:
        order = [int(truck_id) for truck_id in msg.data]
        if len(order) != self.truck_count:
            return

        roles = ['Leader', 'Mid', 'Tail']
        for rank, truck_id in enumerate(order):
            label = roles[min(rank, len(roles) - 1)]
            self.truck_roles[truck_id] = label

    def _update_ui(self) -> None:
        self.dashboard.update(
            self.states,
            self.truck_roles,
            self.maneuver_elapsed_sec,
            self.last_maneuver_duration_sec,
            self.maneuver_count,
        )


class EnergyDashboardUI:
    def __init__(self, root: tk.Tk, truck_count: int = 3):
        self.root = root
        self.root.title("Truck Energy Dashboard")
        self.root.configure(bg="#F3F6FA")
        self.truck_count = truck_count
        self.max_power_kw = 250.0

        self.cards: Dict[int, Dict[str, tk.Widget]] = {}
        self._build_layout()

    def _build_layout(self) -> None:
        header = tk.Frame(self.root, bg="#F3F6FA")
        header.pack(fill=tk.X, padx=14, pady=(10, 6))

        title = tk.Label(
            header,
            text="Electric Truck Energy Monitor",
            font=("Helvetica", 18, "bold"),
            bg="#F3F6FA",
            fg="#132238",
        )
        title.pack(side=tk.LEFT)

        summary_frame = tk.Frame(header, bg="#F3F6FA")
        summary_frame.pack(side=tk.RIGHT)

        self.summary_label = tk.Label(
            summary_frame,
            text="Fleet Efficiency: 0.0 Wh/km",
            font=("Helvetica", 12, "bold"),
            bg="#F3F6FA",
            fg="#2B4A6F",
        )
        self.summary_label.pack(anchor="e")

        self.fleet_soc_label = tk.Label(
            summary_frame,
            text="Fleet SOC Avg: 0.0 % | Sum: 0.0 %",
            font=("Helvetica", 11, "bold"),
            bg="#F3F6FA",
            fg="#2B4A6F",
        )
        self.fleet_soc_label.pack(anchor="e", pady=(4, 0))

        self.maneuver_count_label = tk.Label(
            summary_frame,
            text="Maneuver Count: 0",
            font=("Helvetica", 11),
            bg="#F3F6FA",
            fg="#4A5568",
        )
        self.maneuver_count_label.pack(anchor="e", pady=(4, 0))

        self.maneuver_status_label = tk.Label(
            summary_frame,
            text="Maneuver: Idle",
            font=("Helvetica", 11),
            bg="#F3F6FA",
            fg="#4A5568",
        )
        self.maneuver_status_label.pack(anchor="e", pady=(4, 0))

        self.maneuver_duration_label = tk.Label(
            summary_frame,
            text="Last Maneuver: 0.0 s",
            font=("Helvetica", 11),
            bg="#F3F6FA",
            fg="#4A5568",
        )
        self.maneuver_duration_label.pack(anchor="e")

        body = tk.Frame(self.root, bg="#F3F6FA")
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        for truck_id in range(self.truck_count):
            card = tk.Frame(body, bg="white", bd=1, relief=tk.SOLID, padx=10, pady=10)
            card.grid(row=0, column=truck_id, padx=6, pady=6, sticky="nsew")
            body.grid_columnconfigure(truck_id, weight=1)

            tk.Label(
                card,
                text=f"Truck {truck_id}",
                font=("Helvetica", 15, "bold"),
                bg="white",
                fg="#17375E",
            ).pack(anchor="w")

            role_label = tk.Label(card, text="Role: Leader", font=("Helvetica", 11, "bold"), bg="white", fg="#2B4A6F")
            role_label.pack(anchor="w", pady=(4, 2))

            speed_label = tk.Label(card, text="Speed: 0.0 km/h", font=("Helvetica", 12), bg="white")
            speed_label.pack(anchor="w", pady=(8, 2))

            soc_label = tk.Label(card, text="SOC: 0.0 %", font=("Helvetica", 12), bg="white")
            soc_label.pack(anchor="w", pady=2)

            power_label = tk.Label(card, text="Power: 0.0 kW", font=("Helvetica", 12), bg="white")
            power_label.pack(anchor="w", pady=2)

            whkm_label = tk.Label(card, text="Efficiency: 0.0 Wh/km", font=("Helvetica", 12), bg="white")
            whkm_label.pack(anchor="w", pady=(2, 8))

            tk.Label(card, text="SOC", font=("Helvetica", 10), bg="white", fg="#4A5568").pack(anchor="w")
            soc_canvas = tk.Canvas(card, width=220, height=16, bg="#E6ECF2", highlightthickness=0)
            soc_canvas.pack(anchor="w", pady=(2, 8))

            tk.Label(card, text="Power Usage", font=("Helvetica", 10), bg="white", fg="#4A5568").pack(anchor="w")
            power_canvas = tk.Canvas(card, width=220, height=16, bg="#E6ECF2", highlightthickness=0)
            power_canvas.pack(anchor="w")

            self.cards[truck_id] = {
                "role_label": role_label,
                "speed_label": speed_label,
                "soc_label": soc_label,
                "power_label": power_label,
                "whkm_label": whkm_label,
                "soc_canvas": soc_canvas,
                "power_canvas": power_canvas,
            }

    def update(
        self,
        states: Dict[int, TruckEnergyState],
        truck_roles: Dict[int, str],
        maneuver_elapsed_sec: float,
        last_maneuver_duration_sec: float,
        maneuver_count: int,
    ) -> None:
        total_whkm = 0.0
        total_soc = 0.0
        for truck_id in range(self.truck_count):
            st = states[truck_id]
            speed_kmh = st.speed_ms * 3.6
            power_kw = st.power_w / 1000.0
            total_whkm += st.wh_per_km
            total_soc += st.soc

            card = self.cards[truck_id]
            card["role_label"].config(text=f"Role: {truck_roles.get(truck_id, 'Unknown')}")
            card["speed_label"].config(text=f"Speed: {speed_kmh:5.1f} km/h")
            card["soc_label"].config(text=f"SOC: {st.soc:5.1f} %")
            card["power_label"].config(text=f"Power: {power_kw:6.1f} kW")
            card["whkm_label"].config(text=f"Efficiency: {st.wh_per_km:6.1f} Wh/km")

            self._draw_bar(card["soc_canvas"], min(max(st.soc / 100.0, 0.0), 1.0), "#1AAE6F")
            power_ratio = min(max(abs(power_kw) / self.max_power_kw, 0.0), 1.0)
            power_color = "#D64545" if power_kw >= 0.0 else "#2E86DE"
            self._draw_bar(card["power_canvas"], power_ratio, power_color)

        avg = total_whkm / max(1, self.truck_count)
        avg_soc = total_soc / max(1, self.truck_count)
        self.summary_label.config(text=f"Fleet Efficiency: {avg:.1f} Wh/km")
        self.fleet_soc_label.config(text=f"Fleet SOC Avg: {avg_soc:.1f} % | Sum: {total_soc:.1f} %")
        self.maneuver_count_label.config(text=f"Maneuver Count: {maneuver_count}")

        if maneuver_elapsed_sec > 0.0:
            self.maneuver_status_label.config(text=f"Maneuver Running: {maneuver_elapsed_sec:.1f} s")
        else:
            self.maneuver_status_label.config(text="Maneuver: Idle")

        self.maneuver_duration_label.config(text=f"Last Maneuver: {last_maneuver_duration_sec:.1f} s")

    @staticmethod
    def _draw_bar(canvas: tk.Canvas, ratio: float, color: str) -> None:
        canvas.delete("all")
        full_w = int(canvas["width"])
        bar_w = int(full_w * ratio)
        canvas.create_rectangle(0, 0, full_w, 16, fill="#E6ECF2", width=0)
        canvas.create_rectangle(0, 0, bar_w, 16, fill=color, width=0)


def main(args=None) -> None:
    rclpy.init(args=args)
    root = tk.Tk()
    ui = EnergyDashboardUI(root, truck_count=3)
    node = EnergySubscriber(ui, truck_count=3)

    def shutdown():
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", shutdown)
    signal.signal(signal.SIGINT, lambda _sig, _frame: shutdown())

    def spin_ros():
        if rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            root.after(30, spin_ros)

    root.after(30, spin_ros)
    root.mainloop()


if __name__ == "__main__":
    main()
