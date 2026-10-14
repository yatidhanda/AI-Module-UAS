from __future__ import annotations

import re
from typing import Any

from garuda_edge.ingestion.schemas import ConfidenceResult, GemmaDecision, PolicyResult
from garuda_edge.llm_analyst.prompts import ALLOWED_ACTIONS

BLOCKED_KEYWORDS = [
    "engage",
    "fire",
    "attack",
    "strike",
    "neutralize",
    "weapon",
    "lethal",
    "kill",
    "shoot",
    "launch",
    "destroy",
]

# Whole words and their inflections, so "disengaged", "skilled" or "photoshoot" are not
# flagged while "engaged", "firing" or "destroyed" still are.
_BLOCKED_PATTERN = re.compile(
    r"\b(?:"
    r"engag(?:e|es|ed|ing|ement|ements)"
    r"|fir(?:e|es|ed|ing)"
    r"|attack(?:s|ed|ing|er|ers)?"
    r"|strik(?:e|es|ing)"
    r"|neutrali[sz](?:e|es|ed|ing|ation)"
    r"|weapon(?:s|ry|ized|ised)?"
    r"|lethal(?:ity|ly)?"
    r"|kill(?:s|ed|ing|er|ers)?"
    r"|shoot(?:s|ing|er|ers)?"
    r"|launch(?:es|ed|ing|er|ers)?"
    r"|destroy(?:s|ed|ing|er|ers)?"
    r")\b",
    re.IGNORECASE,
)
# Everyday phrases containing a blocked word (several are detector classes). Only the
# exact phrase is exempt: "open fire" next to a fire hydrant is still blocked.
_BENIGN_PHRASES = re.compile(
    r"\bfire[ _-]?(?:hydrants?|trucks?|engines?|stations?|departments?|escapes?|lanes?|alarms?|fighters?)\b",
    re.IGNORECASE,
)


class PolicyEngine:
    def __init__(self, allowed_actions: list[str] | None = None):
        self.allowed_actions = set(allowed_actions or ALLOWED_ACTIONS)

    def validate(self, decision: GemmaDecision, confidence: ConfidenceResult | None = None) -> PolicyResult:
        corrections: list[str] = []
        final = decision.model_copy(deep=True)

        if _contains_blocked_language(final.model_dump(mode="json")):
            final.decision = "notify_operator"
            final.summary = "Policy gate corrected unsupported analyst language; event requires human review."
            final.recommended_actions = ["notify_operator", "store_evidence"]
            final.risk_flags = ["policy_corrected_blocked_language"]
            corrections.append("blocked_unsafe_action_replaced")
            corrections.append("blocked_language_replaced")

        if final.decision not in self.allowed_actions:
            corrections.append(f"invalid_decision:{final.decision}")
            final.decision = "notify_operator"

        cleaned_actions = [action for action in final.recommended_actions if action in self.allowed_actions]
        if len(cleaned_actions) != len(final.recommended_actions):
            corrections.append("unsupported_actions_removed")
        if not cleaned_actions:
            cleaned_actions = ["notify_operator", "store_evidence"]
            corrections.append("fallback_actions_inserted")
        final.recommended_actions = cleaned_actions
        self._validate_nested_next_step(final, corrections)

        if final.human_review_required is not True:
            corrections.append("human_review_required_forced_true")
            final.human_review_required = True

        if confidence and confidence.level == "low" and final.decision not in {"mark_low_confidence", "request_sensor_confirmation"}:
            final.decision = "mark_low_confidence"
            if "mark_low_confidence" not in final.recommended_actions:
                final.recommended_actions.insert(0, "mark_low_confidence")
            if "request_sensor_confirmation" not in final.recommended_actions:
                final.recommended_actions.append("request_sensor_confirmation")
            corrections.append("low_confidence_forced_safe_decision")

        return PolicyResult(
            allowed=True,
            corrected=bool(corrections),
            corrections=corrections,
            final_decision=final,
        )

    def clean_action_list(self, actions: list[Any]) -> tuple[list[str], bool]:
        cleaned: list[str] = []
        changed = False
        for action in actions:
            action_str = str(action)
            if _contains_blocked_language(action_str):
                cleaned.append("notify_operator")
                changed = True
                continue
            if action_str in self.allowed_actions:
                cleaned.append(action_str)
            else:
                changed = True
        deduped: list[str] = []
        for action in cleaned:
            if action not in deduped:
                deduped.append(action)
        return deduped, changed or len(deduped) != len(cleaned)

    def validate_operator_action(self, action: str) -> tuple[str, list[str]]:
        if _contains_blocked_language(action) or action not in self.allowed_actions:
            return "notify_operator", ["blocked_unsafe_action_replaced"]
        return action, []

    def _validate_nested_next_step(self, final: GemmaDecision, corrections: list[str]) -> None:
        next_step = final.next_step_recommendation
        if not isinstance(next_step, dict):
            return
        primary = str(next_step.get("primary_action") or "")
        if _contains_blocked_language(primary) or primary not in self.allowed_actions:
            next_step["primary_action"] = "notify_operator"
            corrections.append(f"invalid_next_step_primary_action:{primary or 'missing'}")
        supporting = next_step.get("supporting_actions")
        if isinstance(supporting, list):
            cleaned, changed = self.clean_action_list(supporting)
            if changed:
                corrections.append("unsupported_next_step_actions_removed")
            next_step["supporting_actions"] = cleaned or ["store_evidence"]
        final.next_step_recommendation = next_step


def _contains_blocked_language(payload: Any) -> bool:
    if isinstance(payload, dict):
        return any(_contains_blocked_language(value) for value in payload.values())
    if isinstance(payload, list):
        return any(_contains_blocked_language(item) for item in payload)
    if payload is None:
        return False
    return bool(_BLOCKED_PATTERN.search(_BENIGN_PHRASES.sub(" ", str(payload))))
