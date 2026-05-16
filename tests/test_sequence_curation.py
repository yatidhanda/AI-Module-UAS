from pathlib import Path

from garuda_edge.datasets.sequence_curation import score_sequence_for_scenario, write_scenario_manifest
from garuda_edge.datasets.visdrone_video import VisDroneSequenceStats


def stats(**overrides) -> VisDroneSequenceStats:
    base = dict(
        sequence_name="uav-test",
        sequence_dir="/tmp/uav-test",
        annotation_path="/tmp/uav-test.txt",
        frame_count=100,
        annotated_frame_count=100,
        object_count_total=900,
        avg_objects_per_frame=9.0,
        avg_vehicle_count_per_frame=7.0,
        avg_person_count_per_frame=1.0,
        vehicle_track_count=12,
        person_track_count=2,
        class_counts={"car": 650, "van": 50, "pedestrian": 100},
        class_diversity=3,
        median_box_area_ratio=0.001,
        tiny_object_ratio=0.8,
        estimated_camera_motion_score=0.2,
        notes=[],
    )
    base.update(overrides)
    return VisDroneSequenceStats(**base)


def test_vehicle_heavy_scenario_scores_vehicle_sequences():
    scenario = {
        "min_vehicle_count_per_frame_avg": 5,
        "min_vehicle_track_count": 8,
        "prefer_vehicle_ratio": True,
        "priority_classes": ["car", "van", "truck", "bus", "motor"],
    }

    score = score_sequence_for_scenario(stats(), scenario)

    assert score > 50


def test_high_altitude_tiny_vehicle_scenario_scores_tiny_boxes():
    scenario = {
        "min_vehicle_track_count": 5,
        "max_median_box_area_ratio": 0.0025,
        "priority_classes": ["car", "van", "truck", "bus", "motor"],
    }

    tiny_score = score_sequence_for_scenario(stats(median_box_area_ratio=0.001), scenario)
    large_score = score_sequence_for_scenario(stats(median_box_area_ratio=0.02), scenario)

    assert tiny_score > large_score


def test_manifest_writing_creates_json_and_csv(tmp_path: Path):
    item = {
        "scenario_name": "vehicle_heavy_road",
        "sequence_name": "uav-test",
        "sequence_dir": "/tmp/uav-test",
        "annotation_path": "/tmp/uav-test.txt",
        "score": 72.5,
        "stats": stats().to_dict(),
        "recommended_use": "Use for detection, tracking, confidence scoring, evidence logging, and human review.",
        "notes": [],
    }

    json_path, csv_path = write_scenario_manifest("vehicle_heavy_road", [item], tmp_path)

    assert json_path.exists()
    assert csv_path.exists()
    assert "uav-test" in json_path.read_text(encoding="utf-8")
    assert "vehicle_heavy_road" in csv_path.read_text(encoding="utf-8")
