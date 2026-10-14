from __future__ import annotations

from bisect import bisect_left
from pathlib import Path
from typing import Any

import pandas as pd

from garuda_edge.ingestion.schemas import TelemetryPacket


def load_telemetry(path: str | Path, **srt_options: Any) -> list[TelemetryPacket]:
    """Load telemetry from a GarudaEdge CSV or a DJI flight-subtitle .SRT file."""
    if Path(path).suffix.lower() == ".srt":
        from garuda_edge.ingestion.dji_srt import parse_dji_srt

        return parse_dji_srt(path, **srt_options)
    return load_telemetry_csv(path)


def load_telemetry_csv(path: str | Path) -> list[TelemetryPacket]:
    df = pd.read_csv(path)
    return [TelemetryPacket(**row.dropna().to_dict()) for _, row in df.iterrows()]


def get_nearest_telemetry(timestamp_s: float, packets: list[TelemetryPacket]) -> TelemetryPacket:
    if not packets:
        raise ValueError("No telemetry packets available")
    return min(packets, key=lambda packet: abs(packet.timestamp_s - timestamp_s))


class TelemetryTimeline:
    """Telemetry sorted by time, with O(log n) lookup and optional interpolation."""

    def __init__(self, packets: list[TelemetryPacket]):
        if not packets:
            raise ValueError("No telemetry packets available")
        self.packets = sorted(packets, key=lambda packet: packet.timestamp_s)
        self.times = [packet.timestamp_s for packet in self.packets]

    def nearest(self, timestamp_s: float) -> TelemetryPacket:
        i = bisect_left(self.times, timestamp_s)
        if i == 0:
            return self.packets[0]
        if i == len(self.packets):
            return self.packets[-1]
        before, after = self.packets[i - 1], self.packets[i]
        # Ties go to the earlier packet, matching get_nearest_telemetry.
        return after if after.timestamp_s - timestamp_s < timestamp_s - before.timestamp_s else before

    def interpolate(self, timestamp_s: float, max_gap_s: float = 1.0) -> tuple[TelemetryPacket, float]:
        """Packet at timestamp_s and its distance (s) to the nearest real sample.

        Values are interpolated between the two bracketing samples when they are at most
        max_gap_s apart; outside the log or across a larger gap the nearest sample is used.
        """
        i = bisect_left(self.times, timestamp_s)
        if i < len(self.times) and self.times[i] == timestamp_s:
            return self.packets[i], 0.0
        if i == 0 or i == len(self.packets):
            nearest = self.packets[0] if i == 0 else self.packets[-1]
            return nearest, abs(nearest.timestamp_s - timestamp_s)
        before, after = self.packets[i - 1], self.packets[i]
        delta = min(timestamp_s - before.timestamp_s, after.timestamp_s - timestamp_s)
        if after.timestamp_s - before.timestamp_s > max_gap_s:
            return self.nearest(timestamp_s), delta
        return _interpolate_packets(before, after, timestamp_s), delta


_WRAPPED_360_FIELDS = {"yaw_deg", "heading_deg"}


def _interpolate_packets(a: TelemetryPacket, b: TelemetryPacket, timestamp_s: float) -> TelemetryPacket:
    frac = (timestamp_s - a.timestamp_s) / (b.timestamp_s - a.timestamp_s)
    nearer = a if frac < 0.5 else b
    va, vb = a.model_dump(), b.model_dump()
    values: dict[str, Any] = {}
    for key, nearer_value in nearer.model_dump().items():
        x, y = va[key], vb[key]
        if isinstance(x, float) and isinstance(y, float):
            if key.endswith("_deg"):
                value = x + (((y - x) + 180.0) % 360.0 - 180.0) * frac  # shortest arc
                values[key] = value % 360.0 if key in _WRAPPED_360_FIELDS else value
            else:
                values[key] = x + (y - x) * frac
        else:
            values[key] = nearer_value  # categorical, integer or missing: take the nearer sample
    values["timestamp_s"] = timestamp_s
    return TelemetryPacket(**values)
