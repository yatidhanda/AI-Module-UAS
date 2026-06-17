from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GarudaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FramePacket(GarudaModel):
    frame_id: int
    timestamp_s: float
    width: int
    height: int
    frame_path: str | None = None


class Detection(GarudaModel):
    frame_id: int
    timestamp_s: float
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4)


class Track(GarudaModel):
    track_id: str
    class_name: str
    last_bbox_xyxy: list[float] = Field(min_length=4, max_length=4)
    last_confidence: float = Field(ge=0.0, le=1.0)
    track_age_frames: int
    track_stability: float = Field(ge=0.0, le=1.0)
    last_seen_timestamp_s: float


class TelemetryPacket(GarudaModel):
    timestamp_s: float
    lat: float
    lon: float
    alt_m: float
    altitude_m: float | None = None
    heading_deg: float | None = None
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    speed_mps: float | None = None
    airspeed_mps: float | None = None
    groundspeed_mps: float | None = None
    gimbal_pan_deg: float
    gimbal_tilt_deg: float
    battery_pct: float
    packet_loss: float
    latency_ms: float
    link_quality: str | None = None
    vibration: str
    gps_quality: str
    mission_phase: str
    sensor_health: str
    compute_temp_c: float | None = None
    dropped_frames: int | None = None


class GeoEstimate(GarudaModel):
    estimated_lat: float
    estimated_lon: float
    geo_uncertainty_m: float
    method: str


class ConfidenceResult(GarudaModel):
    final_confidence: float = Field(ge=0.0, le=1.0)
    level: str
    reasons: list[str]
    degraded_flags: list[str]


class GemmaDecision(GarudaModel):
    decision: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    priority: str
    summary: str
    recommended_actions: list[str]
    risk_flags: list[str]
    human_review_required: bool
    mission_completion_estimate: dict[str, Any] | None = None
    next_step_recommendation: dict[str, Any] | None = None
    geo_lock_readiness: dict[str, Any] | None = None
    mission_health: dict[str, Any] | None = None
    threat_summary: dict[str, Any] | None = None


class PolicyResult(GarudaModel):
    allowed: bool
    corrected: bool
    corrections: list[str]
    final_decision: GemmaDecision


class ISREvent(GarudaModel):
    event_id: str
    event_type: str
    timestamp_s: float
    frame_id: int
    track_id: str
    object_class: str
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4)
    detection_confidence: float = Field(ge=0.0, le=1.0)
    track_age_frames: int
    track_stability: float = Field(ge=0.0, le=1.0)
    platform: dict[str, Any]
    platform_state: dict[str, Any] | None = None
    link: dict[str, Any]
    geo_estimate: GeoEstimate
    confidence: ConfidenceResult
    analyst_model: str | None = None
    analyst_model_fallback_used: bool = False
    analyst_latency_ms: float | None = None
    gemma_decision: GemmaDecision | None = None
    policy_result: PolicyResult | None = None
    threat_assessment: dict[str, Any] | None = None
    geo_lock_readiness: dict[str, Any] | None = None
    mission_status: dict[str, Any] | None = None
    evidence_frame_path: str | None = None
    audit_notes: list[str]
