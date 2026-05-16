from garuda_edge.threat.alerting import build_threat_alerts


def test_high_threat_alerts_generated_and_sorted():
    events = [
        {
            "event_id": "E1",
            "track_id": "T-low",
            "frame_id": 5,
            "timestamp_s": 1.0,
            "object_class": "car",
            "threat_assessment": {"score": 20, "level": "low", "recommended_operator_actions": []},
        },
        {
            "event_id": "E2",
            "track_id": "T-high",
            "frame_id": 6,
            "timestamp_s": 2.0,
            "object_class": "truck",
            "threat_assessment": {
                "score": 65,
                "level": "high",
                "rationale": "Human review required.",
                "recommended_operator_actions": ["notify_operator"],
            },
        },
        {
            "event_id": "E3",
            "track_id": "T-critical",
            "frame_id": 4,
            "timestamp_s": 3.0,
            "object_class": "jammer",
            "threat_assessment": {
                "score": 90,
                "level": "critical",
                "rationale": "Human review required.",
                "recommended_operator_actions": ["escalate_to_c2"],
            },
        },
    ]

    alerts = build_threat_alerts(events, threshold=60, max_alerts=8)

    assert [alert.track_id for alert in alerts] == ["T-critical", "T-high"]
    assert alerts[0].human_review_required is True
