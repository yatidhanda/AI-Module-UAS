import pytest

from garuda_edge.ingestion.schemas import ConfidenceResult, GemmaDecision
from garuda_edge.policy_engine.policy import BLOCKED_KEYWORDS, PolicyEngine


def decision(**kwargs):
    base = dict(
        decision="notify_operator",
        priority="medium",
        summary="Review event.",
        recommended_actions=["notify_operator"],
        risk_flags=[],
        human_review_required=True,
    )
    base.update(kwargs)
    return GemmaDecision(**base)


def test_invalid_decision_corrected():
    result = PolicyEngine().validate(decision(decision="invalid"))
    assert result.corrected
    assert result.final_decision.decision == "notify_operator"


def test_human_review_forced_true():
    result = PolicyEngine().validate(decision(human_review_required=False))
    assert result.final_decision.human_review_required is True


def test_blocked_words_corrected():
    result = PolicyEngine().validate(decision(summary="engage and destroy target"))
    assert result.final_decision.decision == "notify_operator"
    assert "blocked_language_replaced" in result.corrections


def test_low_confidence_forces_safe_decision():
    conf = ConfidenceResult(final_confidence=0.2, level="low", reasons=[], degraded_flags=["cue_only"])
    result = PolicyEngine().validate(decision(decision="continue_observation"), conf)
    assert result.final_decision.decision == "mark_low_confidence"


def test_policy_blocks_unsafe_nested_next_step_action():
    result = PolicyEngine().validate(
        decision(
            next_step_recommendation={
                "primary_action": "launch",
                "rationale": "unsafe wording",
                "supporting_actions": ["store_evidence", "destroy"],
            }
        )
    )

    assert result.final_decision.next_step_recommendation["primary_action"] == "notify_operator"
    assert "blocked_unsafe_action_replaced" in result.corrections
    assert "destroy" not in result.final_decision.next_step_recommendation["supporting_actions"]


@pytest.mark.parametrize(
    "summary",
    [
        "Fire hydrant observed near the intersection; continue observation.",
        "Fire truck parked beside the road.",
        "Vehicle disengaged from convoy and parked.",
        "Track shows a skilled driver maneuvering.",
        "Photoshoot crew on the sidewalk.",
        "Camera shot of a parked van.",
    ],
)
def test_benign_language_is_not_corrected(summary):
    result = PolicyEngine().validate(decision(summary=summary))
    assert "blocked_language_replaced" not in result.corrections
    assert result.final_decision.summary == summary


@pytest.mark.parametrize(
    "summary",
    [
        "Recommend engaging the vehicle.",
        "Unit fired on the convoy.",
        "Target destroyed.",
        "Open fire near the fire hydrant.",  # an allowed phrase must not hide an unsafe one
        "KILL CONFIRMED",
        "weaponized drone launched",
        "neutralise the threat",
    ],
)
def test_unsafe_language_and_inflections_are_corrected(summary):
    result = PolicyEngine().validate(decision(summary=summary))
    assert "blocked_language_replaced" in result.corrections


def test_every_listed_keyword_is_blocked():
    for keyword in BLOCKED_KEYWORDS:
        result = PolicyEngine().validate(decision(summary=f"Analyst suggests {keyword} now."))
        assert "blocked_language_replaced" in result.corrections, keyword
