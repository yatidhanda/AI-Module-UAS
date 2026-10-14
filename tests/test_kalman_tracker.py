import json

import numpy as np

from garuda_edge.ingestion.schemas import Detection
from garuda_edge.tracking.kalman import BoxKalmanFilter
from garuda_edge.tracking.track_stream import TrackStateStream
from garuda_edge.tracking.tracker import ObjectTracker, _iou

FPS = 30.0


def _box(x: float, y: float, size: float = 40.0) -> list[float]:
    return [x, y, x + size, y + size]


def _det(bbox: list[float], frame_id: int, class_name: str = "car") -> Detection:
    return Detection(
        frame_id=frame_id,
        timestamp_s=frame_id / FPS,
        class_id=0,
        class_name=class_name,
        confidence=0.9,
        bbox_xyxy=bbox,
    )


def _center_x(bbox: list[float]) -> float:
    return (bbox[0] + bbox[2]) / 2.0


def test_filter_converges_to_constant_velocity():
    kf = BoxKalmanFilter(_box(100, 100), 0.0)
    for i in range(1, 31):
        t = i / FPS
        kf.predict(t)
        kf.update(_box(100 + 300 * t, 100))
    vx, vy = kf.velocity_px_s()
    assert abs(vx - 300) / 300 < 0.05
    assert abs(vy) < 15


def test_peek_does_not_change_filter_state():
    kf = BoxKalmanFilter(_box(100, 100), 0.0)
    kf.predict(1 / FPS)
    kf.update(_box(110, 100))
    x, P, t = kf.x.copy(), kf.P.copy(), kf.t
    for k in range(100):
        kf.peek(t + k / 1000.0)
    assert np.array_equal(kf.x, x)
    assert np.array_equal(kf.P, P)
    assert kf.t == t


def test_track_keeps_id_through_short_occlusion():
    tracker = ObjectTracker()
    track_id = None
    for frame in range(10):
        tracks = tracker.update([_det(_box(100 + 20 * frame, 100), frame)], frame, frame / FPS)
        track_id = track_id or tracks[0].track_id
    for frame in range(10, 13):
        assert tracker.update([], frame, frame / FPS) == []
    tracks = tracker.update([_det(_box(100 + 20 * 13, 100), 13)], 13, 13 / FPS)

    # No overlap with the last seen box, so matching on last-seen position alone would re-ID.
    assert _iou(_box(100 + 20 * 9, 100), _box(100 + 20 * 13, 100)) == 0.0
    assert [t.track_id for t in tracks] == [track_id]
    assert tracks[0].velocity_px_s[0] > 0


def test_head_on_crossing_objects_keep_their_ids():
    # 15 px/frame toward each other on the same line: the last-seen-box matcher swaps or re-creates IDs here.
    tracker = ObjectTracker()
    ids = {}
    seen: set[str] = set()
    for frame in range(27):
        a = _det(_box(100 + 15 * frame, 100), frame)
        b = _det(_box(500 - 15 * frame, 100), frame)
        detections = [a, b] if frame % 2 else [b, a]
        tracks = tracker.update(detections, frame, frame / FPS)
        seen |= {t.track_id for t in tracks}
        if frame == 0:
            ids = {"a": next(t.track_id for t in tracks if _center_x(t.last_bbox_xyxy) < 300)}
            ids["b"] = next(t.track_id for t in tracks if t.track_id != ids["a"])
    by_id = {t.track_id: t for t in tracks}
    assert seen == set(ids.values())
    assert _center_x(by_id[ids["a"]].last_bbox_xyxy) > 400  # A ended on the right
    assert _center_x(by_id[ids["b"]].last_bbox_xyxy) < 200  # B ended on the left


def test_detection_of_another_class_never_matches():
    tracker = ObjectTracker()
    car = tracker.update([_det(_box(100, 100), 0, "car")], 0, 0.0)[0]
    tracks = tracker.update([_det(_box(100, 100), 1, "person")], 1, 1 / FPS)
    assert len(tracks) == 1
    assert tracks[0].track_id != car.track_id
    assert tracks[0].class_name == "person"


def test_stream_writes_1000hz_ticks_without_gaps(tmp_path):
    tracker = ObjectTracker()
    path = tmp_path / "track_states.jsonl"
    with TrackStateStream(path, 1000) as stream:
        for frame in range(10):
            t = frame / FPS
            tracker.update([_det(_box(100 + 20 * frame, 100), frame)], frame, t)
            stream.emit_interval(tracker, t, t + 1 / FPS, frame)
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert stream.ticks_written == len(rows) == 334  # ticks 0.000 .. 0.333
    times = [row["t"] for row in rows]
    assert np.allclose(np.diff(times), 0.001)

    last_frame = [row for row in rows if row["frame_id"] == 9]
    assert 33 <= len(last_frame) <= 34
    xs = [_center_x(row["tracks"][0]["bbox_xyxy"]) for row in last_frame]
    assert all(b > a for a, b in zip(xs, xs[1:]))


def test_stream_disabled_at_zero_rate(tmp_path):
    tracker = ObjectTracker()
    tracker.update([_det(_box(100, 100), 0)], 0, 0.0)
    path = tmp_path / "track_states.jsonl"
    stream = TrackStateStream(path, 0)
    assert stream.emit_interval(tracker, 0.0, 1 / FPS, 0) == 0
    stream.close()
    assert not path.exists()
