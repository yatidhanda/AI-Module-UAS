from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT = "data/telemetry/demo_telemetry.csv"


def project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def generate(frames: int, fps: float, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp_s",
        "lat",
        "lon",
        "alt_m",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "gimbal_pan_deg",
        "gimbal_tilt_deg",
        "battery_pct",
        "packet_loss",
        "latency_ms",
        "vibration",
        "gps_quality",
        "mission_phase",
        "sensor_health",
        "compute_temp_c",
        "dropped_frames",
    ]
    lat, lon = 28.6139, 77.2090
    with output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for frame in range(frames):
            t = frame / fps
            lat += random.uniform(-0.000003, 0.000006)
            lon += random.uniform(-0.000004, 0.000006)
            spike = random.random() < 0.025
            writer.writerow(
                {
                    "timestamp_s": round(t, 3),
                    "lat": round(lat, 8),
                    "lon": round(lon, 8),
                    "alt_m": round(random.uniform(70, 120), 2),
                    "roll_deg": round(random.uniform(-3, 3), 2),
                    "pitch_deg": round(random.uniform(-4, 4), 2),
                    "yaw_deg": round((45 + t * 1.5 + random.uniform(-2, 2)) % 360, 2),
                    "gimbal_pan_deg": round(random.uniform(-6, 6), 2),
                    "gimbal_tilt_deg": round(random.uniform(-55, -35), 2),
                    "battery_pct": round(max(15, 100 - t * 0.02), 2),
                    "packet_loss": round(random.uniform(0.12, 0.35) if spike else random.uniform(0, 0.06), 3),
                    "latency_ms": round(random.uniform(300, 650) if spike else random.uniform(80, 150), 1),
                    "vibration": random.choices(["low", "medium", "high"], weights=[0.8, 0.16, 0.04])[0],
                    "gps_quality": random.choices(["good", "medium", "poor"], weights=[0.86, 0.11, 0.03])[0],
                    "mission_phase": random.choice(["area_search", "route_surveillance", "observation"]),
                    "sensor_health": random.choices(["nominal", "degraded"], weights=[0.93, 0.07])[0],
                    "compute_temp_c": round(random.uniform(48, 78), 1),
                    "dropped_frames": random.choices([0, 1, 2, 3], weights=[0.91, 0.06, 0.02, 0.01])[0],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_path = project_path(args.output)
    generate(args.frames, args.fps, output_path)
    print(f"Wrote telemetry to {output_path.resolve()}")


if __name__ == "__main__":
    main()
