from __future__ import annotations

import argparse
import json
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
from garuda_edge.geolocation.projection import project_bbox_to_ground
from garuda_edge.ingestion.schemas import FramePacket, GemmaDecision
from garuda_edge.ingestion.sync import sync_frame_to_telemetry
from garuda_edge.ingestion.telemetry import load_telemetry_csv
from garuda_edge.ingestion.video_reader import open_video
from garuda_edge.llm_analyst.analyst import GemmaAnalyst
from garuda_edge.llm_analyst.ollama_client import SAFE_FALLBACK_DECISION
from garuda_edge.mission.mission_status import compute_mission_status, evaluate_geo_lock_readiness
from garuda_edge.perception.detector import YOLODetector
from garuda_edge.policy_engine.policy import PolicyEngine
from garuda_edge.storage.event_store import EventStore
from garuda_edge.threat.threat_library import load_threat_library
from garuda_edge.threat.threat_scoring import score_threat_for_track
from garuda_edge.tracking.tracker import ObjectTracker
from garuda_edge.utils.visualization import draw_tracks

TELEMETRY_HELP = (
    "Telemetry file not found. Run: python scripts/generate_synthetic_telemetry.py "
    "--frames 1000 --fps 30 --output data/telemetry/demo_telemetry.csv"
)


def project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def load_yaml(path: str) -> dict:
    p = project_path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


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
    args = parser.parse_args()

    video_path = project_path(args.video)
    telemetry_path = project_path(args.telemetry)
    output_dir = project_path(args.output)
    if not telemetry_path.exists():
        print(f"{TELEMETRY_HELP}\nExpected path: {telemetry_path}", file=sys.stderr)
        raise SystemExit(2)

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

    telemetry = load_telemetry_csv(telemetry_path)
    detector = YOLODetector(model_cfg.get("yolo_model", "yolov8n.pt"), model_cfg.get("yolo_conf_threshold", 0.25))
    tracker = ObjectTracker()
    fusion = FusionEngine()
    analyst = GemmaAnalyst()
    policy = PolicyEngine()
    store = EventStore(output_dir)
    camera = CameraModel(**camera_cfg) if camera_cfg else CameraModel()

    gemma_enabled = args.use_gemma.lower() in {"true", "1", "yes"}
    gemma_interval = int(model_cfg.get("gemma_call_interval_frames", 60))
    gemma_state: dict = {}
    gemma_calls = 0
    detections_count = 0
    confidence_values: list[float] = []
    latest_decisions: dict[str, GemmaDecision] = {}
    latest_analyst_metadata: dict[str, dict[str, object]] = {}
    event_history: list[dict[str, object]] = []
    latest_platform_state: dict[str, object] | None = None
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
        sync = sync_frame_to_telemetry(timestamp_s, telemetry)

        for track in tracks:
            geo = project_bbox_to_ground(track.last_bbox_xyxy, width, height, sync.telemetry, camera, sync.sync_delta_s)
            conf = score_confidence(track, track.last_confidence, sync.telemetry, geo, sync.sync_delta_s)
            confidence_values.append(conf.final_confidence)
            evidence_path = None
            if track.track_age_frames == 1 or conf.level == "high":
                evidence_path = str(evidence_dir / f"{track.track_id}_frame_{frame_id:06d}.jpg")
                cv2.imwrite(evidence_path, frame)
            event = fusion.create_event(frame_packet, track, sync.telemetry, geo, conf, sync.sync_delta_s, evidence_path)
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
            if call_gemma:
                decision = analyst.analyze_event(event)
                analyst_metadata = {
                    "model": analyst.model_used,
                    "fallback_used": analyst.fallback_used,
                    "latency_ms": analyst.latency_ms,
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
                    },
                )
            latest_decisions[track.track_id] = decision
            latest_analyst_metadata[track.track_id] = analyst_metadata
            result = policy.validate(decision, conf)
            event.analyst_model = str(analyst_metadata["model"])
            event.analyst_model_fallback_used = bool(analyst_metadata["fallback_used"])
            event.analyst_latency_ms = analyst_metadata["latency_ms"]
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
    elapsed = max(0.001, time.time() - start)
    summary = {
        "frames_processed": frame_id,
        "detections_count": detections_count,
        "tracks_count": len(tracker.all_tracks()),
        "average_fps": round(frame_id / elapsed, 2),
        "average_confidence": round(sum(confidence_values) / len(confidence_values), 3) if confidence_values else 0.0,
        "gemma_calls": gemma_calls,
        "events_path": str(output_dir / "events.jsonl"),
        "annotated_video_path": str(out_video),
        "latest_platform_state": latest_platform_state,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
