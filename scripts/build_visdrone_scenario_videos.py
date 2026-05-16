from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from garuda_edge.datasets.sequence_video_builder import build_mp4_from_sequence


def project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", default="data/sample_events/visdrone_video_manifests")
    parser.add_argument("--output-dir", default="data/input_videos/visdrone")
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--resize-width", type=int, default=None)
    args = parser.parse_args()

    manifest_dir = project_path(args.manifest_dir)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, object]] = {}

    for manifest_path in sorted(manifest_dir.glob("*.json")):
        if manifest_path.name == "manifest_summary.json":
            continue
        scenario_name = manifest_path.stem
        ranked = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not ranked:
            print(f"Skipping {scenario_name}: empty manifest")
            continue
        selected = next((item for item in ranked if Path(item["sequence_dir"]).exists()), ranked[0])
        sequence_dir = Path(selected["sequence_dir"])
        if not sequence_dir.exists():
            print(f"Skipping {scenario_name}: missing sequence directory {sequence_dir}")
            continue
        output_path = output_dir / f"{scenario_name}.mp4"
        build_mp4_from_sequence(
            sequence_dir=sequence_dir,
            output_path=output_path,
            fps=args.fps,
            max_frames=args.max_frames,
            resize_width=args.resize_width,
        )
        summary[scenario_name] = {
            "video_path": str(output_path),
            "sequence_name": selected["sequence_name"],
            "sequence_dir": str(sequence_dir),
            "score": selected["score"],
            "max_frames": args.max_frames,
            "fps": args.fps,
        }
        print(f"Built {output_path} from {selected['sequence_name']}")

    summary_path = output_dir / "manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
