import json

from garuda_edge.storage.event_store import EventStore


def _event(i, track_id, confidence=0.8, lat=28.6):
    return {
        "event_id": f"E{i}",
        "track_id": track_id,
        "object_class": "car",
        "timestamp_s": i / 30,
        "frame_id": i,
        "track_age_frames": i + 1,
        "track_stability": 0.9,
        "confidence": {"level": "high", "final_confidence": confidence},
        "geo_estimate": {"estimated_lat": lat, "estimated_lon": 77.2, "geo_uncertainty_m": 3.0},
        "platform_state": {"lat": 28.6 + i * 1e-6, "lon": 77.2, "timestamp_s": i / 30, "frame_id": i},
        "policy_result": {"final_decision": {"decision": f"decision_{i}"}},
    }


def test_derived_files_written_on_flush_and_match_disk_derivation(tmp_path):
    store = EventStore(tmp_path, flush_interval_s=3600)
    for i in range(50):
        store.append_event(_event(i, f"T{i % 3}", confidence=0.5 + i / 100))
    assert not (tmp_path / "tracks.json").exists()  # throttled: no per-event rewrite

    store.flush()
    tracks = json.loads((tmp_path / "tracks.json").read_text())
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert tracks == store.read_tracks()
    assert summary == store._summary_from_events()
    assert summary["total_events"] == 50 and summary["active_tracks"] == 3
    assert summary["latest_decision"] == "decision_49"
    assert summary["latest_platform_state"]["frame_id"] == 49


def test_zero_interval_flushes_every_append(tmp_path):
    store = EventStore(tmp_path, flush_interval_s=0)
    store.append_event(_event(0, "T0"))
    assert json.loads((tmp_path / "summary.json").read_text())["total_events"] == 1


def test_new_writer_continues_from_existing_log(tmp_path):
    first = EventStore(tmp_path)
    for i in range(5):
        first.append_event(_event(i, "T0"))
    first.flush()
    second = EventStore(tmp_path, flush_interval_s=0)
    second.append_event(_event(5, "T1"))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["total_events"] == 6 and summary["active_tracks"] == 2
