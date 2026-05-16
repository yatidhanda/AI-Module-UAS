from pathlib import Path

from garuda_edge.datasets import visdrone_video
from garuda_edge.datasets.visdrone_video import VisDroneSequenceInfo, compute_sequence_stats, parse_mot_annotation


def test_mot_parser_converts_classes_and_ignores_regions(tmp_path: Path):
    annotation = tmp_path / "seq.txt"
    annotation.write_text(
        "\n".join(
            [
                "1,10,10,20,30,40,1,4,0,0",
                "1,11,0,0,100,100,1,0,0,0",
                "2,12,5,6,7,8,1,1,0,0",
                "bad,row",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_mot_annotation(annotation)

    assert len(parsed[1].objects) == 1
    assert parsed[1].objects[0].class_id == 3
    assert parsed[1].objects[0].target_id == 10
    assert parsed[2].objects[0].class_id == 0


def test_sequence_stats_uses_annotations_and_mocked_frame_size(tmp_path: Path, monkeypatch):
    sequence_dir = tmp_path / "VisDrone2019-MOT-train" / "sequences" / "uav0001"
    sequence_dir.mkdir(parents=True)
    for frame_id in range(1, 11):
        (sequence_dir / f"{frame_id:07d}.jpg").write_text("", encoding="utf-8")

    annotation = tmp_path / "VisDrone2019-MOT-train" / "annotations" / "uav0001.txt"
    annotation.parent.mkdir(parents=True)
    rows = []
    for frame_id in range(1, 11):
        rows.append(f"{frame_id},{frame_id},10,20,20,20,1,4,0,0")
        rows.append(f"{frame_id},{frame_id + 100},30,40,10,10,1,1,0,0")
    annotation.write_text("\n".join(rows), encoding="utf-8")

    monkeypatch.setattr(visdrone_video, "load_frame_size", lambda frame_path: (1000, 1000))
    monkeypatch.setattr(visdrone_video, "_estimate_camera_motion", lambda frames, sample_every_n_frames, notes: 0.2)

    stats = compute_sequence_stats(
        VisDroneSequenceInfo(
            sequence_name="uav0001",
            sequence_dir=sequence_dir,
            annotation_path=annotation,
            sequence_style="mot",
            frame_count=10,
        )
    )

    assert stats.frame_count == 10
    assert stats.object_count_total == 20
    assert stats.avg_vehicle_count_per_frame == 1.0
    assert stats.avg_person_count_per_frame == 1.0
    assert stats.vehicle_track_count == 10
    assert stats.class_counts["car"] == 10
    assert stats.median_box_area_ratio > 0
