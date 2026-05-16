from __future__ import annotations

from pathlib import Path

import pandas as pd

from garuda_edge.ingestion.schemas import TelemetryPacket


def load_telemetry_csv(path: str | Path) -> list[TelemetryPacket]:
    df = pd.read_csv(path)
    return [TelemetryPacket(**row.dropna().to_dict()) for _, row in df.iterrows()]


def get_nearest_telemetry(timestamp_s: float, packets: list[TelemetryPacket]) -> TelemetryPacket:
    if not packets:
        raise ValueError("No telemetry packets available")
    return min(packets, key=lambda packet: abs(packet.timestamp_s - timestamp_s))
