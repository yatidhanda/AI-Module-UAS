from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from garuda_edge.storage.event_store import EventStore

app = FastAPI(title="GarudaEdge ISR Fusion Module API")
store = EventStore()


class OperatorAction(BaseModel):
    event_id: str
    track_id: str | None = None
    action: str
    operator_id: str | None = None
    notes: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "module": "garuda_edge"}


@app.get("/events")
def events(limit: int = 100) -> list[dict[str, Any]]:
    return store.read_events(limit=limit)


@app.get("/events/latest")
def latest_event() -> dict[str, Any]:
    latest = store.read_latest_event()
    if not latest:
        raise HTTPException(status_code=404, detail="No events found")
    return latest


@app.get("/tracks")
def tracks() -> list[dict[str, Any]]:
    return store.read_tracks()


@app.get("/tracks/{track_id}")
def track(track_id: str) -> dict[str, Any]:
    for item in store.read_tracks():
        if item["track_id"] == track_id:
            return item
    raise HTTPException(status_code=404, detail="Track not found")


@app.get("/summary")
def summary() -> dict[str, Any]:
    return store.read_summary()


@app.post("/operator-action")
def operator_action(action: OperatorAction) -> dict[str, str]:
    path = Path("data/outputs/operator_actions.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(action.model_dump()) + "\n")
    return {"status": "logged"}
