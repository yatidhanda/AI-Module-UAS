from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from garuda_edge.ingestion.schemas import GemmaDecision, ISREvent
from garuda_edge.llm_analyst.ollama_client import OllamaClient, SAFE_FALLBACK_DECISION
from garuda_edge.llm_analyst.prompts import build_analyst_prompt


class GemmaAnalyst:
    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()

    def analyze_event(self, event: ISREvent | dict[str, Any]) -> GemmaDecision:
        event_dict = event.model_dump(mode="json") if hasattr(event, "model_dump") else dict(event)
        prompt = build_analyst_prompt(event_dict)
        decision = self.client.generate_json(prompt)
        try:
            return GemmaDecision(**decision)
        except ValidationError as exc:
            self.client.last_error = f"Analyst response failed schema validation: {exc}"
            return GemmaDecision(**SAFE_FALLBACK_DECISION)

    @property
    def model_used(self) -> str:
        return self.client.last_model_used or self.client.model

    @property
    def fallback_used(self) -> bool:
        return self.client.last_fallback_used

    @property
    def latency_ms(self) -> float | None:
        return self.client.last_latency_ms
