from __future__ import annotations

from garuda_edge.ingestion.schemas import ConfidenceResult, GeoEstimate, TelemetryPacket, Track


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def score_confidence(
    track: Track,
    detection_confidence: float,
    telemetry: TelemetryPacket,
    geo_estimate: GeoEstimate,
    sync_delta_s: float,
    high_threshold: float = 0.75,
    medium_threshold: float = 0.5,
) -> ConfidenceResult:
    reasons: list[str] = []
    flags: list[str] = []

    age_score = clamp(track.track_age_frames / 30.0)
    score = (
        detection_confidence * 0.34
        + track.track_stability * 0.22
        + age_score * 0.10
        + 0.34
    )

    reasons.append(f"detection_confidence={detection_confidence:.2f}")
    reasons.append(f"track_stability={track.track_stability:.2f}")
    reasons.append(f"track_age_frames={track.track_age_frames}")

    gps_penalty = {"good": 0.0, "medium": 0.08, "poor": 0.18}.get(telemetry.gps_quality, 0.12)
    if gps_penalty:
        flags.append(f"gps_{telemetry.gps_quality}")
    vib_penalty = {"low": 0.0, "medium": 0.07, "high": 0.16}.get(telemetry.vibration, 0.08)
    if vib_penalty:
        flags.append(f"vibration_{telemetry.vibration}")
    score -= gps_penalty + vib_penalty

    if telemetry.packet_loss > 0.1:
        penalty = min(0.18, telemetry.packet_loss)
        score -= penalty
        flags.append("packet_loss_high")
    if telemetry.latency_ms > 300:
        score -= min(0.14, (telemetry.latency_ms - 300.0) / 1000.0)
        flags.append("latency_high")
    if geo_estimate.geo_uncertainty_m > 60:
        score -= min(0.20, (geo_estimate.geo_uncertainty_m - 60.0) / 200.0)
        flags.append("geo_uncertainty_high")
    if telemetry.sensor_health != "nominal":
        score -= 0.12
        flags.append("sensor_degraded")
    if (telemetry.dropped_frames or 0) > 0:
        score -= min(0.10, (telemetry.dropped_frames or 0) * 0.02)
        flags.append("dropped_frames")
    if sync_delta_s > 0.5:
        score -= min(0.10, sync_delta_s * 0.05)
        flags.append("sync_delta_high")

    final = clamp(score)
    if final >= high_threshold:
        level = "high"
    elif final >= medium_threshold:
        level = "medium"
    else:
        level = "low"
        flags.append("cue_only")

    if flags:
        reasons.append("degraded by " + ", ".join(sorted(set(flags))))
    else:
        reasons.append("stable detection, track, link, and platform telemetry")

    return ConfidenceResult(
        final_confidence=round(final, 3),
        level=level,
        reasons=reasons,
        degraded_flags=sorted(set(flags)),
    )
