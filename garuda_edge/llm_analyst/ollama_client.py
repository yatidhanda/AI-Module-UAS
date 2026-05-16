from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests
import yaml

try:
    from loguru import logger
except Exception:
    import logging

    logger = logging.getLogger(__name__)


SAFE_FALLBACK_DECISION: dict[str, Any] = {
    "decision": "notify_operator",
    "confidence": 0.0,
    "priority": "medium",
    "summary": "Analyst model unavailable; event requires human review.",
    "recommended_actions": ["notify_operator", "store_evidence"],
    "risk_flags": ["llm_unavailable"],
    "human_review_required": True,
    "mission_completion_estimate": None,
    "next_step_recommendation": {
        "primary_action": "notify_operator",
        "rationale": "Analyst model unavailable; operator review is required.",
        "supporting_actions": ["store_evidence"],
    },
    "geo_lock_readiness": None,
    "mission_health": {
        "status": "needs_review",
        "perception_health": "unknown",
        "tracking_health": "unknown",
        "geo_confidence_health": "unknown",
        "link_health": "unknown",
        "analyst_health": "unavailable",
        "key_degradations": ["llm_unavailable"],
    },
    "threat_summary": None,
}


class OllamaClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        fallback_models: list[str] | None = None,
        timeout_s: int | float | None = None,
        config_path: str | Path | None = None,
    ):
        cfg = self._load_config(config_path)
        configured_fallbacks = cfg.get("ollama_fallback_models") or []
        self.base_url = str(base_url or cfg.get("ollama_base_url", "http://localhost:11434")).rstrip("/")
        self.model = model or str(cfg.get("ollama_model", "gemma4:e4b"))
        self.temperature = float(temperature if temperature is not None else cfg.get("gemma_temperature", 0.1))
        self.fallback_models = list(fallback_models if fallback_models is not None else configured_fallbacks)
        self.timeout_s = float(timeout_s if timeout_s is not None else cfg.get("ollama_timeout_s", 120))
        self.last_model_used: str | None = None
        self.last_fallback_used = False
        self.last_latency_ms: float | None = None
        self.last_raw_response: str | None = None
        self.last_error: str | None = None
        self.last_attempted_models: list[str] = []

    @staticmethod
    def _load_config(config_path: str | Path | None = None) -> dict[str, Any]:
        path = Path(config_path) if config_path else Path(__file__).resolve().parents[2] / "configs" / "model.yaml"
        if not path.exists():
            return {}
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def generate_json(self, prompt: str, timeout: int | float | None = None) -> dict[str, Any]:
        timeout_s = float(timeout if timeout is not None else self.timeout_s)
        self.last_model_used = None
        self.last_fallback_used = False
        self.last_latency_ms = None
        self.last_raw_response = None
        self.last_error = None
        self.last_attempted_models = []

        errors: list[str] = []
        for candidate in self._model_attempt_order():
            self.last_attempted_models.append(candidate)
            start = time.perf_counter()
            result, raw, error = self._try_generate(prompt, candidate, timeout_s)
            self.last_latency_ms = round((time.perf_counter() - start) * 1000, 2)
            if result is not None:
                self.last_model_used = candidate
                self.last_fallback_used = candidate != self.model
                self.last_raw_response = raw
                self.last_error = None
                return result
            errors.append(error)

        self.last_model_used = self.model
        self.last_fallback_used = False
        self.last_raw_response = json.dumps(SAFE_FALLBACK_DECISION)
        self.last_error = "; ".join(errors)
        logger.warning(self.last_error)
        return dict(SAFE_FALLBACK_DECISION)

    def _model_attempt_order(self) -> list[str]:
        ordered: list[str] = []
        for model in [self.model, *self.fallback_models]:
            if model and model not in ordered:
                ordered.append(model)
        return ordered

    def _try_generate(self, prompt: str, model: str, timeout_s: float) -> tuple[dict[str, Any] | None, str | None, str]:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
        }
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=timeout_s)
            if response.status_code == 404:
                return None, None, self._model_missing_message(model, response)
            response.raise_for_status()
            outer = response.json()
            raw = outer.get("response", "")
            if isinstance(raw, dict):
                return raw, json.dumps(raw), ""
            return json.loads(raw), raw, ""
        except requests.exceptions.ConnectionError:
            return None, None, f"Ollama is not running at {self.base_url}; start it with 'ollama serve'."
        except requests.exceptions.Timeout:
            return None, None, f"Ollama request to {model} timed out after {timeout_s:g}s."
        except requests.exceptions.RequestException as exc:
            return None, None, f"Ollama request failed for {model}: {exc}"
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return None, None, f"Ollama model {model} did not return valid JSON: {exc}"
        except Exception as exc:
            return None, None, f"Ollama request failed for {model}: {exc}"

    def _model_missing_message(self, model: str, response: requests.Response) -> str:
        detail = response.text.strip()
        if model == self.model:
            return f"Ollama model {model} is not pulled; run 'ollama pull {model}'. {detail}"
        return f"Ollama fallback model {model} is not pulled; run 'ollama pull {model}'. {detail}"
