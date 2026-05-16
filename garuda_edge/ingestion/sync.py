from __future__ import annotations

from dataclasses import dataclass

try:
    from loguru import logger
except Exception:
    import logging

    logger = logging.getLogger(__name__)

from garuda_edge.ingestion.schemas import TelemetryPacket
from garuda_edge.ingestion.telemetry import get_nearest_telemetry


@dataclass(frozen=True)
class SyncResult:
    telemetry: TelemetryPacket
    sync_delta_s: float
    warning: str | None = None


def sync_frame_to_telemetry(
    frame_timestamp_s: float, telemetry_packets: list[TelemetryPacket], threshold_s: float = 0.5
) -> SyncResult:
    telemetry = get_nearest_telemetry(frame_timestamp_s, telemetry_packets)
    delta = abs(telemetry.timestamp_s - frame_timestamp_s)
    warning = None
    if delta > threshold_s:
        warning = f"sync_delta_exceeds_threshold:{delta:.3f}s"
        logger.warning(warning)
    return SyncResult(telemetry=telemetry, sync_delta_s=delta, warning=warning)
