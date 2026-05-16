from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from garuda_edge.datasets.visdrone_video import VisDroneSequenceStats, compute_sequence_stats, discover_visdrone_sequences


DEFAULT_MANIFEST_DIR = Path("data/sample_events/visdrone_video_manifests")


def score_sequence_for_scenario(stats: VisDroneSequenceStats | dict[str, Any], scenario_config: dict[str, Any]) -> float:
    data = _stats_dict(stats)
    score = 0.0

    score += _min_score(data.get("avg_vehicle_count_per_frame", 0.0), scenario_config.get("min_vehicle_count_per_frame_avg"), 22)
    score += _min_score(data.get("vehicle_track_count", 0), scenario_config.get("min_vehicle_track_count"), 18)
    score += _min_score(data.get("avg_person_count_per_frame", 0.0), scenario_config.get("min_person_count_per_frame_avg"), 22)
    score += _min_score(data.get("avg_objects_per_frame", 0.0), scenario_config.get("min_total_objects_per_frame_avg"), 18)
    score += _min_score(data.get("estimated_camera_motion_score", 0.0), scenario_config.get("min_camera_motion_score"), 20)

    score += _max_score(data.get("avg_vehicle_count_per_frame", 0.0), scenario_config.get("max_vehicle_count_per_frame_avg"), 12)
    score += _max_score(data.get("median_box_area_ratio", 0.0), scenario_config.get("max_median_box_area_ratio"), 22)

    if scenario_config.get("prefer_vehicle_ratio"):
        total = max(float(data.get("avg_objects_per_frame", 0.0)), 0.001)
        score += min(12.0, 12.0 * float(data.get("avg_vehicle_count_per_frame", 0.0)) / total)

    if scenario_config.get("require_class_diversity"):
        score += 12.0 if int(data.get("class_diversity", 0)) >= 3 else 4.0 * int(data.get("class_diversity", 0))

    priority_classes = scenario_config.get("priority_classes") or []
    if priority_classes:
        class_counts = data.get("class_counts") or {}
        total_objects = max(int(data.get("object_count_total", 0)), 1)
        priority_count = sum(int(class_counts.get(class_name, 0)) for class_name in priority_classes)
        score += min(14.0, 14.0 * priority_count / total_objects)

    if data.get("frame_count", 0):
        score += min(5.0, float(data["frame_count"]) / 60.0)

    return round(score, 4)


def curate_sequences_for_scenarios(
    dataset_root: Path,
    scenario_config_path: Path,
    limit_per_scenario: int,
) -> dict[str, list[dict[str, Any]]]:
    scenarios = yaml.safe_load(Path(scenario_config_path).read_text(encoding="utf-8")) or {}
    sequence_infos = discover_visdrone_sequences(Path(dataset_root))
    stats_by_name = [compute_sequence_stats(sequence_info) for sequence_info in sequence_infos]

    curated: dict[str, list[dict[str, Any]]] = {}
    for scenario_name, scenario_config in scenarios.items():
        ranked: list[dict[str, Any]] = []
        for stats in stats_by_name:
            score = score_sequence_for_scenario(stats, scenario_config)
            ranked.append(_manifest_item(scenario_name, score, stats, scenario_config))
        curated[scenario_name] = sorted(ranked, key=lambda item: item["score"], reverse=True)[:limit_per_scenario]
    return curated


def write_scenario_manifest(scenario_name: str, ranked_sequences: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{scenario_name}.json"
    csv_path = output_dir / f"{scenario_name}.csv"

    json_path.write_text(json.dumps(ranked_sequences, indent=2), encoding="utf-8")

    fieldnames = ["scenario_name", "sequence_name", "sequence_dir", "annotation_path", "score", "recommended_use", "notes", "stats_json"]
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for item in ranked_sequences:
            writer.writerow(
                {
                    "scenario_name": item["scenario_name"],
                    "sequence_name": item["sequence_name"],
                    "sequence_dir": item["sequence_dir"],
                    "annotation_path": item["annotation_path"],
                    "score": item["score"],
                    "recommended_use": item["recommended_use"],
                    "notes": ";".join(item.get("notes") or []),
                    "stats_json": json.dumps(item["stats"], sort_keys=True),
                }
            )
    return json_path, csv_path


def _manifest_item(
    scenario_name: str,
    score: float,
    stats: VisDroneSequenceStats,
    scenario_config: dict[str, Any],
) -> dict[str, Any]:
    stats_payload = stats.to_dict()
    notes = list(stats.notes)
    if score <= 0:
        notes.append("low_scenario_match_score")
    return {
        "scenario_name": scenario_name,
        "sequence_name": stats.sequence_name,
        "sequence_dir": stats.sequence_dir,
        "annotation_path": stats.annotation_path,
        "score": score,
        "stats": stats_payload,
        "recommended_use": _recommended_use(scenario_name, scenario_config),
        "notes": notes,
    }


def _recommended_use(scenario_name: str, scenario_config: dict[str, Any]) -> str:
    description = scenario_config.get("description", scenario_name.replace("_", " "))
    return f"{description} Use for detection, tracking, confidence scoring, evidence logging, and human review."


def _stats_dict(stats: VisDroneSequenceStats | dict[str, Any]) -> dict[str, Any]:
    if isinstance(stats, dict):
        return stats
    if hasattr(stats, "to_dict"):
        return stats.to_dict()
    if is_dataclass(stats):
        return asdict(stats)
    raise TypeError(f"Unsupported stats type: {type(stats)!r}")


def _min_score(value: float, threshold: float | int | None, weight: float) -> float:
    if threshold is None:
        return 0.0
    threshold = float(threshold)
    if threshold <= 0:
        return weight
    return min(weight, weight * max(0.0, float(value)) / threshold)


def _max_score(value: float, threshold: float | int | None, weight: float) -> float:
    if threshold is None:
        return 0.0
    threshold = float(threshold)
    if threshold <= 0:
        return 0.0
    value = max(0.0, float(value))
    if value <= threshold:
        return weight
    return max(0.0, weight * (threshold / value))
