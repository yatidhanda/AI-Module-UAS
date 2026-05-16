from garuda_edge.ingestion.schemas import ConfidenceResult, GemmaDecision
from garuda_edge.policy_engine.policy import PolicyEngine


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
