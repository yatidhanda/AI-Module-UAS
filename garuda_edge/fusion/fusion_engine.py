from __future__ import annotations

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
            platform={
                "lat": telemetry.lat,
                "lon": telemetry.lon,
                "alt_m": telemetry.alt_m,
                "roll_deg": telemetry.roll_deg,
                "pitch_deg": telemetry.pitch_deg,
                "yaw_deg": telemetry.yaw_deg,
                "gimbal_pan_deg": telemetry.gimbal_pan_deg,
                "gimbal_tilt_deg": telemetry.gimbal_tilt_deg,
                "battery_pct": telemetry.battery_pct,
                "mission_phase": telemetry.mission_phase,
                "sensor_health": telemetry.sensor_health,
            },
            link={
                "packet_loss": telemetry.packet_loss,
                "latency_ms": telemetry.latency_ms,
                "sync_delta_s": sync_delta_s,
            },
            geo_estimate=geo_estimate,
            confidence=confidence,
            evidence_frame_path=evidence_frame_path,
            audit_notes=notes,
        )
