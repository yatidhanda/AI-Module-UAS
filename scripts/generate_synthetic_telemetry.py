from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from garuda_edge.ingestion.synthetic_telemetry import PROFILES, generate_rows, write_telemetry_csv

DEFAULT_OUTPUT = "data/telemetry/demo_telemetry.csv"


def project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", choices=PROFILES, default="hover", help="hover: hold position; transect: fly straight legs")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible telemetry")
    args = parser.parse_args()
    output_path = project_path(args.output)
    write_telemetry_csv(generate_rows(args.frames, args.fps, args.profile, args.seed), output_path)
    print(f"Wrote telemetry to {output_path.resolve()}")


if __name__ == "__main__":
    main()
