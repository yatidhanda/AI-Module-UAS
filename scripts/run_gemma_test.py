from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from garuda_edge.ingestion.schemas import ConfidenceResult, GemmaDecision, GeoEstimate, ISREvent
from garuda_edge.llm_analyst.ollama_client import OllamaClient, SAFE_FALLBACK_DECISION
from garuda_edge.llm_analyst.prompts import build_analyst_prompt
from garuda_edge.mission.mission_status import compute_mission_status, evaluate_geo_lock_readiness
from garuda_edge.policy_engine.policy import PolicyEngine
from garuda_edge.threat.threat_library import load_threat_library
from garuda_edge.threat.threat_scoring import score_threat_for_track


def load_model_config() -> dict:
    path = PROJECT_ROOT / "configs" / "model.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_mission_config() -> dict:
    path = PROJECT_ROOT / "configs" / "mission.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def sample_event() -> ISREvent:
    return ISREvent(
        event_id="EVT-GEMMA-TEST-0001",
        event_type="track_update",
        timestamp_s=42.5,
        frame_id=1275,
        track_id="T-0042",
        object_class="person",
        bbox_xyxy=[612.0, 338.0, 681.0, 491.0],
        detection_confidence=0.71,
        track_age_frames=86,
        track_stability=0.64,
        platform={
            "lat": 34.05231,
            "lon": -118.24372,
            "alt_m": 118.0,
            "roll_deg": 1.2,
            "pitch_deg": -4.8,
            "yaw_deg": 92.0,
            "gimbal_pan_deg": 4.0,
            "gimbal_tilt_deg": -31.0,
            "battery_pct": 68.0,
            "mission_phase": "area_monitoring",
            "sensor_health": "nominal",
        },
        link={"packet_loss": 0.03, "latency_ms": 82.0, "sync_delta_s": 0.08},
        geo_estimate=GeoEstimate(
            estimated_lat=34.05266,
            estimated_lon=-118.24291,
            geo_uncertainty_m=21.5,
            method="camera_ray_ground_plane",
        ),
        confidence=ConfidenceResult(
            final_confidence=0.62,
            level="medium",
            reasons=["stable track", "moderate detector confidence", "nominal sensor health"],
            degraded_flags=[],
        ),
        analyst_model=None,
        audit_notes=["synthetic analyst test packet; human review required"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    cfg = load_model_config()
    client = OllamaClient(
        model=args.model or cfg.get("ollama_model"),
        base_url=cfg.get("ollama_base_url"),
        temperature=cfg.get("gemma_temperature"),
        fallback_models=cfg.get("ollama_fallback_models"),
        timeout_s=cfg.get("ollama_timeout_s"),
    )
    event = sample_event()
    mission = load_mission_config()
    threat_library = load_threat_library("configs/threat_library.yaml")
    event_payload = event.model_dump(mode="json")
    event.threat_assessment = score_threat_for_track(event_payload, mission, threat_library).model_dump(mode="json")
    event_payload = event.model_dump(mode="json")
    event.geo_lock_readiness = evaluate_geo_lock_readiness(event_payload, mission)
    event_payload = event.model_dump(mode="json")
    event.mission_status = compute_mission_status([event_payload], [], {}, mission, "data/outputs")
    prompt = build_analyst_prompt(event.model_dump(mode="json"))
    parsed = client.generate_json(prompt)

    print("Raw model output:")
    print(client.last_raw_response or json.dumps(parsed))
    print()

    print("Parsed JSON:")
    print(json.dumps(parsed, indent=2))
    print()

    try:
        decision = GemmaDecision(**parsed)
    except ValidationError as exc:
        print(f"Schema validation correction: {exc}")
        fallback = dict(SAFE_FALLBACK_DECISION)
        fallback["risk_flags"] = ["llm_schema_invalid"]
        decision = GemmaDecision(**fallback)
    policy_result = PolicyEngine().validate(decision, event.confidence)

    print("Policy-checked output:")
    print(json.dumps(policy_result.model_dump(mode="json"), indent=2))
    print()

    print("Analyst runtime:")
    print(
        json.dumps(
            {
                "model_used": client.last_model_used,
                "fallback_used": client.last_fallback_used,
                "latency_ms": client.last_latency_ms,
                "attempted_models": client.last_attempted_models,
                "error": client.last_error,
            },
            indent=2,
        )
    )

    if client.last_error and parsed.get("risk_flags") == ["llm_unavailable"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
