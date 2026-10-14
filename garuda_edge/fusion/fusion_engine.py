from __future__ import annotations

import math
from typing import Any

from garuda_edge.ingestion.schemas import ConfidenceResult, FramePacket, GeoEstimate, ISREvent, TelemetryPacket, Track
from garuda_edge.utils.ids import make_event_id


class FusionEngine:
    def create_event(
        self,
        frame: FramePacket,
        track: Track,
        telemetry: TelemetryPacket,
        geo_estimate: GeoEstimate,
        confidence: ConfidenceResult,
        sync_delta_s: float,
        evidence_frame_path: str | None = None,
    ) -> ISREvent:
        notes = ["geo-track lock means persistent tracking with approximate geolocation for human review"]
        if sync_delta_s > 0.5:
            notes.append(f"telemetry sync delta high: {sync_delta_s:.3f}s")
        platform_state = _platform_state_from_telemetry(telemetry, frame)
        return ISREvent(
            event_id=make_event_id(),
            event_type="track_update",
            timestamp_s=frame.timestamp_s,
            frame_id=frame.frame_id,
            track_id=track.track_id,
            object_class=track.class_name,
            bbox_xyxy=track.last_bbox_xyxy,
            detection_confidence=track.last_confidence,
            track_age_frames=track.track_age_frames,
            track_stability=track.track_stability,
            bbox_velocity_px_s=track.velocity_px_s,
            platform={
                "lat": telemetry.lat,
                "lon": telemetry.lon,
                "alt_m": telemetry.alt_m,
                "altitude_m": platform_state["altitude_m"],
                "heading_deg": platform_state["heading_deg"],
                "roll_deg": telemetry.roll_deg,
                "pitch_deg": telemetry.pitch_deg,
                "yaw_deg": telemetry.yaw_deg,
                "speed_mps": platform_state["speed_mps"],
                "gimbal_pan_deg": telemetry.gimbal_pan_deg,
                "gimbal_tilt_deg": telemetry.gimbal_tilt_deg,
                "battery_pct": telemetry.battery_pct,
                "gps_quality": telemetry.gps_quality,
                "mission_phase": telemetry.mission_phase,
                "sensor_health": telemetry.sensor_health,
            },
            platform_state=platform_state,
            link={
                "packet_loss": telemetry.packet_loss,
                "latency_ms": telemetry.latency_ms,
                "link_quality": platform_state["link_quality"],
                "sync_delta_s": sync_delta_s,
            },
            geo_estimate=geo_estimate,
            confidence=confidence,
            evidence_frame_path=evidence_frame_path,
            audit_notes=notes,
        )


def _platform_state_from_telemetry(telemetry: TelemetryPacket, frame: FramePacket) -> dict[str, Any]:
    packet_loss = _float_or_none(getattr(telemetry, "packet_loss", None))
    latency_ms = _float_or_none(getattr(telemetry, "latency_ms", None))
    heading = _first_float(telemetry, "heading_deg", "yaw_deg")
    altitude = _first_float(telemetry, "altitude_m", "alt_m")
    speed = _first_float(telemetry, "speed_mps", "groundspeed_mps", "airspeed_mps")
    link_quality = getattr(telemetry, "link_quality", None) or _derive_link_quality(packet_loss, latency_ms)
    return {
        "lat": _float_or_none(getattr(telemetry, "lat", None)),
        "lon": _float_or_none(getattr(telemetry, "lon", None)),
        "altitude_m": altitude,
        "heading_deg": heading,
        "roll_deg": _float_or_none(getattr(telemetry, "roll_deg", None)),
        "pitch_deg": _float_or_none(getattr(telemetry, "pitch_deg", None)),
        "yaw_deg": _float_or_none(getattr(telemetry, "yaw_deg", None)),
        "speed_mps": speed,
        "battery_pct": _float_or_none(getattr(telemetry, "battery_pct", None)),
        "packet_loss": packet_loss,
        "latency_ms": latency_ms,
        "link_quality": link_quality,
        "gps_quality": getattr(telemetry, "gps_quality", None),
        "sensor_health": getattr(telemetry, "sensor_health", None),
        "timestamp_s": frame.timestamp_s,
        "frame_id": frame.frame_id,
    }


def _first_float(source: Any, *field_names: str) -> float | None:
    for field_name in field_names:
        value = _float_or_none(getattr(source, field_name, None))
        if value is not None:
            return value
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _derive_link_quality(packet_loss: float | None, latency_ms: float | None) -> str:
    if packet_loss is not None and packet_loss > 0.10:
        return "degraded"
    if latency_ms is not None and latency_ms > 300:
        return "degraded"
    return "nominal"
