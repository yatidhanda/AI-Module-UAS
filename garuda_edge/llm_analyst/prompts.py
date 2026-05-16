from __future__ import annotations

import json


ALLOWED_ACTIONS = [
    "continue_observation",
    "store_evidence",
    "notify_operator",
    "request_recheck",
    "escalate_to_c2",
    "mark_false_positive",
    "request_sensor_confirmation",
    "mark_low_confidence",
    "return_to_base",
    "hold_position",
    "end_mission_review",
    "reacquire_track",
    "center_camera_on_track",
]


def build_analyst_prompt(event: dict) -> str:
    return (
        "You are an ISR sensor-fusion analyst for a non-weaponized human-in-the-loop UAS monitoring demo.\n"
        "Return only valid JSON.\n"
        "Do not use markdown.\n"
        "Do not invent objects, identities, intent, hostile behavior, or unsupported hazards.\n"
        "Use only supplied event and mission data.\n"
        "Choose decision only from enum: "
        + ", ".join(ALLOWED_ACTIONS)
        + ".\n"
        "human_review_required must always be true.\n"
        "Recommended actions must be observation, review, logging, C2 notification, mission-control, or sensor-cueing actions only.\n"
        "Do not recommend force, kinetic effects, weapons, or final high-consequence authorization.\n"
        "Threat level means mission/platform risk estimate for human review, not hostile intent.\n"
        "Estimate mission completion only from supplied scenario/output/event summary data.\n"
        "Explain geo-lock readiness using confidence, track age, geolocation uncertainty, and mission target classes.\n"
        "Explain mission health using perception, tracking, geo confidence, link health, and analyst availability.\n"
        "If threat_assessment is present, explain whether risk comes from mission context, platform risk, degraded link, or object category.\n"
        "JSON schema: {"
        "\"decision\": string, "
        "\"confidence\": number, "
        "\"priority\": \"low|medium|high\", "
        "\"summary\": string, "
        "\"recommended_actions\": [string], "
        "\"risk_flags\": [string], "
        "\"human_review_required\": true, "
        "\"mission_completion_estimate\": {\"percent_complete\": number, \"status\": string, \"rationale\": string, \"remaining_gaps\": [string]}, "
        "\"next_step_recommendation\": {\"primary_action\": string, \"rationale\": string, \"supporting_actions\": [string]}, "
        "\"geo_lock_readiness\": {\"status\": string, \"ready_tracks\": [string], \"not_ready_tracks\": [string], \"rationale\": string}, "
        "\"mission_health\": {\"status\": string, \"perception_health\": string, \"tracking_health\": string, \"geo_confidence_health\": string, \"link_health\": string, \"analyst_health\": string, \"key_degradations\": [string]}, "
        "\"threat_summary\": {\"highest_threat_track_id\": string|null, \"highest_threat_level\": string|null, \"rationale\": string, \"high_threat_tracks\": [string]}"
        "}\n"
        "Event data:\n"
        + json.dumps(event, sort_keys=True)
    )
