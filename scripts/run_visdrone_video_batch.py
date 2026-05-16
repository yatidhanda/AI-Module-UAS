from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCENARIOS = [
    "vehicle_heavy_road",
    "urban_intersection",
    "pedestrian_overhead",
    "high_altitude_tiny_vehicle",
    "moving_drifting_camera",
]


def project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default="data/input_videos/visdrone")
    parser.add_argument("--telemetry-dir", default="data/telemetry/visdrone")
    parser.add_argument("--output-root", default="data/outputs/visdrone")
    parser.add_argument("--use-gemma", default="false")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--telemetry-fps", type=float, default=15)
    args = parser.parse_args()

    use_gemma = args.use_gemma.lower() in {"true", "1", "yes"}
    max_frames = args.max_frames if args.max_frames is not None else (60 if use_gemma else 300)
    if use_gemma:
        print("Warning: Gemma 4 E4B can be slow. This batch keeps the existing event-throttled pipeline behavior.")

    video_dir = project_path(args.video_dir)
    telemetry_dir = project_path(args.telemetry_dir)
    output_root = project_path(args.output_root)
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    ran_any = False
    for scenario_name in SCENARIOS:
        video_path = video_dir / f"{scenario_name}.mp4"
        if not video_path.exists():
            print(f"Skipping {scenario_name}: missing {video_path}")
            continue
        ran_any = True
        telemetry_path = telemetry_dir / f"{scenario_name}.csv"
        output_dir = output_root / scenario_name

        _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "generate_synthetic_telemetry.py"),
                "--frames",
                str(max_frames),
                "--fps",
                str(args.telemetry_fps),
                "--output",
                str(telemetry_path),
            ]
        )
        _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
                "--video",
                str(video_path),
                "--telemetry",
                str(telemetry_path),
                "--output",
                str(output_dir),
                "--max-frames",
                str(max_frames),
                "--use-gemma",
                str(use_gemma).lower(),
            ]
        )
        print(f"Scenario output written to {output_dir}")

    if not ran_any:
        print(f"No scenario videos found in {video_dir}. Build them with scripts/build_visdrone_scenario_videos.py.")
        raise SystemExit(2)

    print("Open the Streamlit dashboard and select a VisDrone output folder in the sidebar to compare runs.")


def _run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
