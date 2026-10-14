from __future__ import annotations

from dataclasses import dataclass

try:
    from loguru import logger
except Exception:
    import logging

    logger = logging.getLogger(__name__)

from garuda_edge.ingestion.schemas import TelemetryPacket
from garuda_edge.ingestion.telemetry import TelemetryTimeline


@dataclass(frozen=True)
class SyncResult:
    telemetry: TelemetryPacket
    sync_delta_s: float
    warning: str | None = None


def sync_frame_to_telemetry(
    frame_timestamp_s: float,
    telemetry_packets: list[TelemetryPacket] | TelemetryTimeline,
    threshold_s: float = 0.5,
    interpolate: bool = False,
) -> SyncResult:
    """Telemetry for a frame. sync_delta_s is the distance to the nearest real sample.

    Pass a TelemetryTimeline (built once) for O(log n) lookups on long flight logs.
    """
    timeline = telemetry_packets if isinstance(telemetry_packets, TelemetryTimeline) else TelemetryTimeline(telemetry_packets)
    if interpolate:
        telemetry, delta = timeline.interpolate(frame_timestamp_s)
    else:
        telemetry = timeline.nearest(frame_timestamp_s)
        delta = abs(telemetry.timestamp_s - frame_timestamp_s)
    warning = None
    if delta > threshold_s:
        warning = f"sync_delta_exceeds_threshold:{delta:.3f}s"
        logger.warning(warning)
    return SyncResult(telemetry=telemetry, sync_delta_s=delta, warning=warning)
