from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from garuda_edge.datasets.visdrone_video import summarize_dataset


def project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/datasets/VisDrone")
    args = parser.parse_args()

    dataset_root = project_path(args.dataset_root)
    summary = summarize_dataset(dataset_root)

    print(f"Dataset root: {dataset_root}")
    print(f"Discovered sequences: {summary['sequence_count']}")
    print(f"Annotation files: {summary['annotation_file_count']}")
    print(f"Total frames: {summary['total_frames']}")
    print(f"MOT-style folders found: {summary['mot_style_found']}")
    print(f"VID-style folders found: {summary['vid_style_found']}")
    print()

    if summary["det_only_detected"]:
        print(
            "Only VisDrone DET image folders detected. For video tracking tests, download VisDrone MOT/VID "
            "sequence data or use existing drone videos."
        )
        return

    print("Top 10 longest sequences:")
    for index, sequence in enumerate(summary["top_10_longest_sequences"], start=1):
        print(
            f"{index:02d}. {sequence['sequence_name']} | frames={sequence['frame_count']} | "
            f"style={sequence['sequence_style']} | annotation={sequence['annotation_path']}"
        )

    if not summary["sequence_count"]:
        print("No MOT/VID sequence folders found. Expected image sequences under VisDrone2019-MOT-* or VisDrone2019-VID-*.")
    print()
    print("Summary JSON:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
