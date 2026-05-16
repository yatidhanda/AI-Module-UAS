from pathlib import Path

import cv2
import numpy as np

from garuda_edge.dashboard.track_preview import get_selected_track_preview, render_track_highlight_image


def write_image(path: Path, color: tuple[int, int, int] = (20, 30, 40)) -> None:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[:, :] = color
    cv2.imwrite(str(path), image)


def test_selected_track_preview_prefers_latest_event_evidence(tmp_path: Path):
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    older = evidence_dir / "older.jpg"
    latest = evidence_dir / "latest.jpg"
    write_image(older)
    write_image(latest)
    events = [
        {"event_id": "E1", "track_id": "T1", "frame_id": 1, "timestamp_s": 1, "evidence_frame_path": str(older)},
        {
            "event_id": "E2",
            "track_id": "T1",
            "frame_id": 3,
            "timestamp_s": 3,
            "object_class": "car",
            "bbox_xyxy": [10, 10, 40, 40],
            "evidence_frame_path": str(latest),
        },
    ]

    preview = get_selected_track_preview("T1", events, tmp_path)

    assert preview is not None
    assert Path(preview["image_path"]) == latest
    assert preview["event"]["event_id"] == "E2"
    assert preview["bbox_xyxy"] == [10, 10, 40, 40]


def test_selected_track_preview_falls_back_to_older_evidence(tmp_path: Path):
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    older = evidence_dir / "older.jpg"
    write_image(older)
    events = [
        {
            "event_id": "E1",
            "track_id": "T1",
            "frame_id": 1,
            "timestamp_s": 1,
            "object_class": "car",
            "bbox_xyxy": [10, 10, 40, 40],
            "evidence_frame_path": str(older),
        },
        {
            "event_id": "E2",
            "track_id": "T1",
            "frame_id": 5,
            "timestamp_s": 5,
            "object_class": "car",
            "bbox_xyxy": [20, 20, 50, 50],
        },
    ]

    preview = get_selected_track_preview("T1", events, tmp_path)

    assert preview is not None
    assert Path(preview["image_path"]) == older
    assert preview["event"]["event_id"] == "E2"
    assert preview["image_event"]["event_id"] == "E1"
    assert preview["bbox_xyxy"] == [10, 10, 40, 40]


def test_render_track_highlight_image_creates_output_file(tmp_path: Path):
    image_path = tmp_path / "source.jpg"
    write_image(image_path)

    output_path = Path(
        render_track_highlight_image(
            str(image_path),
            [10, 10, 60, 50],
            tmp_path,
            "T-0004",
            142,
            object_class="car",
        )
    )

    assert output_path.exists()
    assert output_path.parent == tmp_path / "dashboard_frames"
    highlighted = cv2.imread(str(output_path))
    assert highlighted is not None
    assert highlighted[10, 10, 2] > 150


def test_render_track_highlight_image_missing_bbox_does_not_crash(tmp_path: Path):
    image_path = tmp_path / "source.jpg"
    write_image(image_path)

    output_path = Path(render_track_highlight_image(str(image_path), None, tmp_path, "T-0004", 142))

    assert output_path.exists()
    assert output_path.parent == tmp_path / "dashboard_frames"
