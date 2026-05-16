from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from garuda_edge.datasets.sequence_curation import (
    DEFAULT_MANIFEST_DIR,
    curate_sequences_for_scenarios,
    write_scenario_manifest,
)


def project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/datasets/VisDrone")
    parser.add_argument("--scenario-config", default="configs/video_scenarios.yaml")
    parser.add_argument("--output-dir", default=str(DEFAULT_MANIFEST_DIR))
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    curated = curate_sequences_for_scenarios(
        dataset_root=project_path(args.dataset_root),
        scenario_config_path=project_path(args.scenario_config),
        limit_per_scenario=args.limit,
    )

    output_dir = project_path(args.output_dir)
    for scenario_name, ranked in curated.items():
        json_path, csv_path = write_scenario_manifest(scenario_name, ranked, output_dir)
        print(f"\n{scenario_name}")
        print(f"Manifest: {json_path}")
        print(f"CSV: {csv_path}")
        for index, item in enumerate(ranked[:5], start=1):
            stats = item["stats"]
            print(
                f"{index}. score={item['score']:.2f} sequence={item['sequence_name']} "
                f"frames={stats['frame_count']} objects/frame={stats['avg_objects_per_frame']} "
                f"vehicles/frame={stats['avg_vehicle_count_per_frame']} people/frame={stats['avg_person_count_per_frame']} "
                f"motion={stats['estimated_camera_motion_score']}"
            )

    if not curated:
        print("No scenarios were curated. Check configs/video_scenarios.yaml and the dataset root.")


if __name__ == "__main__":
    main()
