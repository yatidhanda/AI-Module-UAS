# Safety and Governance

GarudaEdge is a non-weaponized, human-in-the-loop UAS ISR monitoring demo.

Unsupported actions include force application, kinetic effects, weapons-related control, and final high-consequence authorization. The allowed action space is limited to observation, evidence storage, operator notification, C2 relay, sensor confirmation, low-confidence marking, track reacquisition, camera centering, and mission-control logging.

The policy gate validates every model output. It forces `human_review_required=true`, removes unsupported actions, corrects invalid decisions, blocks unsafe language, and records corrections in audit notes.

Gemma runs only on structured ISR event packets. Events include the analyst model used, whether a fallback model was used, response latency when available, and policy corrections for audit.

The system provides decision support only. It does not provide autonomous high-consequence action, final authorization, or identity determination.

## Mission / Platform Risk Library

The threat/risk library estimates mission and platform risk for human review. A high score means the object or track may matter to the drone mission, link quality, telemetry integrity, geo-track confidence, or operator workload. It is not a hostile-intent finding and it is not an authorization mechanism.

Each event can include `threat_assessment` with score, level, category, rationale, contributing factors, safe recommended operator actions, and `human_review_required=true`. Recommended actions remain limited to observation, review, evidence logging, sensor confirmation, C2 notification, and low-confidence handling.

Geo-lock readiness is deterministic. A track is ready only when mission target class, confidence, track age, and geolocation uncertainty satisfy `configs/mission.yaml`. The dashboard displays readiness as an operator cue for review and evidence packaging.

Gemma may explain mission health, mission completion estimate, next step, geo-lock readiness, and threat/risk context, but the policy gate still validates outputs and forces human review.
