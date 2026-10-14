from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from garuda_edge.confidence.scoring import score_confidence
from garuda_edge.fusion.fusion_engine import FusionEngine
from garuda_edge.geolocation.camera_model import CameraModel
from garuda_edge.geolocation.geo_tracker import GeoTrackFilter
from garuda_edge.geolocation.projection import gps_sigma_m, project_bbox_to_ground
from garuda_edge.ingestion.schemas import FramePacket, GemmaDecision
from garuda_edge.ingestion.sync import sync_frame_to_telemetry
from garuda_edge.ingestion.telemetry import TelemetryTimeline, load_telemetry
from garuda_edge.ingestion.video_reader import open_video
from garuda_edge.llm_analyst.analyst import GemmaAnalyst
from garuda_edge.llm_analyst.ollama_client import SAFE_FALLBACK_DECISION
from garuda_edge.llm_analyst.worker import AnalystWorker
from garuda_edge.mission.mission_status import compute_mission_status, evaluate_geo_lock_readiness
from garuda_edge.perception.detector import YOLODetector
from garuda_edge.policy_engine.policy import PolicyEngine
from garuda_edge.storage.event_store import EventStore
from garuda_edge.threat.threat_library import load_threat_library
from garuda_edge.threat.threat_scoring import score_threat_for_track
from garuda_edge.tracking.track_stream import TrackStateStream
from garuda_edge.tracking.tracker import ObjectTracker
from garuda_edge.utils.visualization import draw_tracks

TELEMETRY_HELP = (
    "Telemetry file not found. Run: python scripts/generate_synthetic_telemetry.py "
    "--frames 1000 --fps 30 --output data/telemetry/demo_telemetry.csv"
)


# Files a run writes into its output folder. operator_actions.jsonl is a human audit log
# and is never removed by --overwrite.
GENERATED_OUTPUTS = [
    "events.jsonl",
    "tracks.json",
    "summary.json",
    "track_states.jsonl",
    "analyst_decisions.jsonl",
    "annotated_demo.mp4",
    "evidence",
]


def project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def load_yaml(path: str) -> dict:
    p = project_path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    """Refuse to mix runs: events.jsonl is append-only and track IDs restart every run."""
    if (output_dir / "events.jsonl").exists():
        if not overwrite:
            print(
                f"Output folder already has a run: {output_dir / 'events.jsonl'}\n"
                "Pass --overwrite to replace it, or choose a different --output.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        for name in GENERATED_OUTPUTS:
            path = output_dir / name
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        if (output_dir / "operator_actions.jsonl").exists():
            print("Kept operator_actions.jsonl from the previous run (human audit log).", file=sys.stderr)


def should_call_gemma(track_id: str, frame_id: int, level: str, state: dict, interval: int, degraded: bool) -> bool:
    prior = state.get(track_id)
    due = prior is None or prior["level"] != level or frame_id - prior["frame_id"] >= interval or degraded
    if due:
        state[track_id] = {"frame_id": frame_id, "level": level}
    return due


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--telemetry", required=True)
    parser.add_argument("--output", default="data/outputs")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--use-gemma", default="true")
    parser.add_argument(
        "--gemma-async",
        action="store_true",
        help="Run Gemma on a background thread so it never stalls the frame loop (for live feeds)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing run in --output")
    parser.add_argument(
        "--track-output-hz",
        type=float,
        default=1000.0,
        help="Rate of Kalman-predicted track states written to track_states.jsonl (0 disables)",
    )
    parser.add_argument(
        "--srt-gimbal-tilt-deg",
        type=float,
        default=-90.0,
        help="Gimbal tilt for DJI .SRT telemetry without gimbal angles (-90 = straight down)",
    )
    parser.add_argument(
        "--srt-yaw-deg",
        type=float,
        default=None,
        help="Camera heading for DJI .SRT telemetry when it cannot be derived from gimbal yaw or motion",
    )
    args = parser.parse_args()

    video_path = project_path(args.video)
    telemetry_path = project_path(args.telemetry)
    output_dir = project_path(args.output)
    if not telemetry_path.exists():
        print(f"{TELEMETRY_HELP}\nExpected path: {telemetry_path}", file=sys.stderr)
        raise SystemExit(2)

    prepare_output_dir(output_dir, args.overwrite)
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = load_yaml("configs/model.yaml")
    camera_cfg = load_yaml("configs/camera.yaml")
    mission_cfg = load_yaml("configs/mission.yaml")
    threat_library = load_threat_library("configs/threat_library.yaml")

    cap = open_video(video_path)
    import cv2

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    out_video = output_dir / "annotated_demo.mp4"
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    telemetry = TelemetryTimeline(
        load_telemetry(telemetry_path, gimbal_tilt_deg=args.srt_gimbal_tilt_deg, yaw_deg=args.srt_yaw_deg)
    )
    detector = YOLODetector(
        model_cfg.get("yolo_model", "yolov8n.pt"),
        model_cfg.get("yolo_conf_threshold", 0.25),
        device=model_cfg.get("yolo_device", "auto"),
    )
    tracker = ObjectTracker()
    geo_filter = GeoTrackFilter()
    fusion = FusionEngine()
    analyst = GemmaAnalyst()
    policy = PolicyEngine()
    store = EventStore(output_dir)
    track_stream = TrackStateStream(output_dir / "track_states.jsonl", args.track_output_hz)
    camera = CameraModel(**camera_cfg) if camera_cfg else CameraModel()

    gemma_enabled = args.use_gemma.lower() in {"true", "1", "yes"}
    worker = AnalystWorker(analyst, output_dir / "analyst_decisions.jsonl") if gemma_enabled and args.gemma_async else None
    gemma_interval = int(model_cfg.get("gemma_call_interval_frames", 60))
    gemma_state: dict = {}
    gemma_calls = 0
    detections_count = 0
    confidence_values: list[float] = []
    latest_decisions: dict[str, GemmaDecision] = {}
    latest_analyst_metadata: dict[str, dict[str, object]] = {}
    event_history: list[dict[str, object]] = []
    latest_platform_state: dict[str, object] | None = None
    evidence_levels: dict[str, str] = {}
    start = time.time()

    frame_id = 0
    while frame_id < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp_s = frame_id / fps
        frame_packet = FramePacket(frame_id=frame_id, timestamp_s=timestamp_s, width=width, height=height)
        detections = detector.predict(frame, frame_id, timestamp_s)
        detections_count += len(detections)
        tracks = tracker.update(detections, frame_id, timestamp_s)
        track_stream.emit_interval(tracker, timestamp_s, timestamp_s + 1.0 / fps, frame_id)
        sync = sync_frame_to_telemetry(timestamp_s, telemetry, interpolate=True)

        geo_filter.prune(t.track_id for t in tracker.all_tracks())
        for track in tracks:
            geo_raw = project_bbox_to_ground(track.last_bbox_xyxy, width, height, sync.telemetry, camera, sync.sync_delta_s)
            geo = geo_filter.update(track.track_id, geo_raw, timestamp_s, gps_sigma_m(sync.telemetry.gps_quality))
            conf = score_confidence(track, track.last_confidence, sync.telemetry, geo, sync.sync_delta_s)
            confidence_values.append(conf.final_confidence)
            evidence_path = None
            # One frame per new track and per confidence-level change, not every high-confidence frame.
            if evidence_levels.get(track.track_id) != conf.level:
                evidence_path = str(evidence_dir / f"{track.track_id}_frame_{frame_id:06d}.jpg")
                cv2.imwrite(evidence_path, frame)
            evidence_levels[track.track_id] = conf.level
            event = fusion.create_event(frame_packet, track, sync.telemetry, geo, conf, sync.sync_delta_s, evidence_path)
            event.geo_raw_estimate = geo_raw
            event_payload = event.model_dump(mode="json")
            event.threat_assessment = score_threat_for_track(event_payload, mission_cfg, threat_library).model_dump(mode="json")
            event_payload = event.model_dump(mode="json")
            event.geo_lock_readiness = evaluate_geo_lock_readiness(event_payload, mission_cfg)
            event_payload = event.model_dump(mode="json")
            recent_events = [*event_history[-500:], event_payload]
            event.mission_status = compute_mission_status(recent_events, [], {}, mission_cfg, output_dir)

            call_gemma = gemma_enabled and should_call_gemma(
                track.track_id, frame_id, conf.level, gemma_state, gemma_interval, bool(conf.degraded_flags)
            )
            finished = None
            if worker is not None:
                if call_gemma:
                    worker.submit(track.track_id, event)
                    gemma_calls += 1
                finished = worker.latest(track.track_id)
            if finished is not None:
                decision, analyst_metadata = finished
            elif call_gemma and worker is None:
                decision = analyst.analyze_event(event)
                analyst_metadata = {
                    "model": analyst.model_used,
                    "fallback_used": analyst.fallback_used,
                    "latency_ms": analyst.latency_ms,
                    "source_event_id": event.event_id,
                }
                gemma_calls += 1
            else:
                decision = latest_decisions.get(track.track_id) or GemmaDecision(**SAFE_FALLBACK_DECISION)
                analyst_metadata = latest_analyst_metadata.get(
                    track.track_id,
                    {
                        "model": model_cfg.get("ollama_model", "gemma4:e4b"),
                        "fallback_used": False,
                        "latency_ms": None,
                        "source_event_id": None,
                    },
                )
            latest_decisions[track.track_id] = decision
            latest_analyst_metadata[track.track_id] = analyst_metadata
            result = policy.validate(decision, conf)
            event.analyst_model = str(analyst_metadata["model"])
            event.analyst_model_fallback_used = bool(analyst_metadata["fallback_used"])
            event.analyst_latency_ms = analyst_metadata["latency_ms"]
            event.analyst_source_event_id = analyst_metadata.get("source_event_id")
            event.gemma_decision = decision
            event.policy_result = result
            if result.corrected:
                event.audit_notes.extend(result.corrections)
            final_event_payload = event.model_dump(mode="json")
            latest_platform_state = final_event_payload.get("platform_state") or latest_platform_state
            store.append_event(event)
            event_history.append(final_event_payload)

        writer.write(draw_tracks(frame, tracks))
        frame_id += 1

    cap.release()
    writer.release()
    track_stream.close()
    store.flush()
    gemma_unfinished = worker.close(timeout_s=float(model_cfg.get("ollama_timeout_s", 120))) if worker else 0
    elapsed = max(0.001, time.time() - start)
    summary = {
        "frames_processed": frame_id,
        "detections_count": detections_count,
        "tracks_count": len(tracker.all_tracks()),
        "average_fps": round(frame_id / elapsed, 2),
        "average_confidence": round(sum(confidence_values) / len(confidence_values), 3) if confidence_values else 0.0,
        "gemma_calls": gemma_calls,
        "gemma_mode": ("async" if worker else "sync") if gemma_enabled else "disabled",
        "gemma_completed": worker.completed if worker else gemma_calls,
        "gemma_coalesced": worker.coalesced if worker else 0,
        "gemma_unfinished": gemma_unfinished,
        "evidence_frames": sum(1 for _ in evidence_dir.glob("*.jpg")),
        "events_path": str(output_dir / "events.jsonl"),
        "annotated_video_path": str(out_video),
        "track_output_hz": track_stream.rate_hz,
        "track_states_path": str(track_stream.path) if track_stream.enabled else None,
        "track_state_ticks": track_stream.ticks_written,
        "latest_platform_state": latest_platform_state,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
