from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from loguru import logger
except Exception:
    import logging

    logger = logging.getLogger(__name__)

from garuda_edge.ingestion.schemas import GemmaDecision, ISREvent
from garuda_edge.llm_analyst.analyst import GemmaAnalyst


class AnalystWorker:
    """Runs Gemma analysis on a background thread so slow LLM calls don't stall the frame loop.

    At most one request is pending per track: a newer submission for the same track
    replaces an older unstarted one (counted in `coalesced`), so the queue cannot build a
    backlog of stale frames. Callers read the latest finished decision with `latest()`;
    each result records which event it analysed. Every finished analysis is also appended
    to `log_path` (when given) for audit. Decisions still go through the policy gate when
    they are attached to events.
    """

    def __init__(self, analyst: GemmaAnalyst, log_path: str | Path | None = None):
        self.analyst = analyst
        self.log_path = Path(log_path) if log_path else None
        self.completed = 0
        self.coalesced = 0
        self._pending: dict[str, dict[str, Any]] = {}
        self._results: dict[str, tuple[GemmaDecision, dict[str, Any]]] = {}
        self._cond = threading.Condition()
        self._busy = False
        self._stopping = False
        self._thread = threading.Thread(target=self._run, name="gemma-analyst", daemon=True)
        self._thread.start()

    def submit(self, track_id: str, event: ISREvent | dict[str, Any]) -> None:
        payload = event.model_dump(mode="json") if hasattr(event, "model_dump") else dict(event)
        with self._cond:
            if self._stopping:
                return
            if track_id in self._pending:
                self.coalesced += 1
                del self._pending[track_id]  # re-insert so it queues behind other tracks
            self._pending[track_id] = payload
            self._cond.notify()

    def latest(self, track_id: str) -> tuple[GemmaDecision, dict[str, Any]] | None:
        with self._cond:
            return self._results.get(track_id)

    def close(self, timeout_s: float = 0.0) -> int:
        """Stop the worker, waiting up to timeout_s for queued work; returns requests left unanalysed."""
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while (self._pending or self._busy) and time.monotonic() < deadline:
                self._cond.wait(timeout=max(0.0, deadline - time.monotonic()))
            dropped = len(self._pending)
            self._pending.clear()
            self._stopping = True
            self._cond.notify_all()
        self._thread.join(timeout=max(0.0, deadline - time.monotonic()) + 1.0)
        if dropped:
            logger.warning(f"{dropped} queued Gemma analyses were not run before shutdown")
        return dropped

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._pending and not self._stopping:
                    self._cond.wait()
                if self._stopping and not self._pending:
                    return
                track_id = next(iter(self._pending))
                payload = self._pending.pop(track_id)
                self._busy = True
            try:
                decision = self.analyst.analyze_event(payload)
                metadata = {
                    "model": self.analyst.model_used,
                    "fallback_used": self.analyst.fallback_used,
                    "latency_ms": self.analyst.latency_ms,
                    "source_event_id": payload.get("event_id"),
                }
                self._record(track_id, decision, metadata)
            except Exception as exc:  # keep the worker alive; the pipeline falls back to the safe decision
                logger.warning(f"Gemma analysis failed for {track_id}: {exc}")
            finally:
                with self._cond:
                    self._busy = False
                    self._cond.notify_all()

    def _record(self, track_id: str, decision: GemmaDecision, metadata: dict[str, Any]) -> None:
        with self._cond:
            self._results[track_id] = (decision, metadata)
            self.completed += 1
        if self.log_path is not None:
            line = {
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "track_id": track_id,
                **metadata,
                "decision": decision.model_dump(mode="json"),
            }
            with self.log_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(line) + "\n")
