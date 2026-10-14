import json
import threading
import time

from garuda_edge.ingestion.schemas import GemmaDecision
from garuda_edge.llm_analyst.ollama_client import SAFE_FALLBACK_DECISION
from garuda_edge.llm_analyst.worker import AnalystWorker


class SlowAnalyst:
    """Stands in for GemmaAnalyst; blocks until released so tests control timing."""

    model_used, fallback_used, latency_ms = "fake-model", False, 1.0

    def __init__(self):
        self.release = threading.Event()
        self.started = threading.Event()
        self.seen: list[str] = []

    def analyze_event(self, payload):
        self.started.set()
        self.release.wait(5)
        self.seen.append(payload["event_id"])
        return GemmaDecision(**{**SAFE_FALLBACK_DECISION, "summary": f"analysed {payload['event_id']}"})


def test_submit_does_not_block_and_newer_requests_replace_queued_ones(tmp_path):
    analyst = SlowAnalyst()
    worker = AnalystWorker(analyst, tmp_path / "analyst_decisions.jsonl")
    start = time.monotonic()
    worker.submit("T1", {"event_id": "E1"})
    assert analyst.started.wait(2)  # E1 is now being analysed
    worker.submit("T1", {"event_id": "E2"})
    worker.submit("T1", {"event_id": "E3"})  # replaces queued E2
    worker.submit("T2", {"event_id": "E4"})
    assert time.monotonic() - start < 1.0
    assert worker.latest("T1") is None  # nothing finished yet

    analyst.release.set()
    assert worker.close(timeout_s=5) == 0
    assert analyst.seen == ["E1", "E3", "E4"]
    assert worker.coalesced == 1 and worker.completed == 3
    decision, metadata = worker.latest("T1")
    assert metadata["source_event_id"] == "E3" and decision.summary == "analysed E3"
    logged = [json.loads(line) for line in (tmp_path / "analyst_decisions.jsonl").read_text().splitlines()]
    assert [row["source_event_id"] for row in logged] == ["E1", "E3", "E4"]


def test_close_reports_work_left_when_timeout_expires():
    analyst = SlowAnalyst()
    worker = AnalystWorker(analyst)
    worker.submit("T1", {"event_id": "E1"})
    analyst.started.wait(2)
    worker.submit("T2", {"event_id": "E2"})
    assert worker.close(timeout_s=0.1) == 1  # E2 never started
    analyst.release.set()
