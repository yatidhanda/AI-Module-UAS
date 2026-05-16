# Architecture

GarudaEdge uses a two-brain architecture.

The fast brain is deterministic perception. YOLO runs on each video frame and produces object detections. The tracker turns detections into object memory with stable track IDs, track age, and track stability. Telemetry sync binds each frame to synthetic drone platform, link, and mission state.

The geo engine estimates approximate position using a simple camera-ray to ground-plane projection. It is a cueing estimate, not survey-grade geolocation. The confidence engine is deterministic and combines detection quality, track stability, track age, GPS quality, vibration, packet loss, latency, sensor health, dropped frames, sync delta, and geolocation uncertainty.

The slow brain is Gemma 4 E4B through Ollama. Gemma is event-driven, not frame-driven. It is called for new tracks, confidence transitions, interval refreshes, or degraded conditions. It explains, summarizes, prioritizes, and recommends safe ISR follow-up actions. If `gemma4:e4b` is unavailable, the analyst client tries `gemma3:4b` and then `gemma3n:e4b` before returning the safe model-unavailable decision.

The policy engine is the safety gate. It validates decision enums, removes unsupported actions, blocks unsafe language, forces `human_review_required=true`, and corrects low-confidence decisions. Dashboard and API layers expose the auditable event stream for operators and C2-style integration.

## Mission-Aware Risk Layer

The fusion path now attaches deterministic mission/platform risk metadata to each ISR event. `configs/threat_library.yaml` defines object profiles, aliases, score bands, context modifiers, and alert thresholds. The scorer combines object class, mission priority classes, watchlist/geo-lock context, track persistence, confidence, geolocation uncertainty, link health, and sensor health.

This layer produces `threat_assessment` and `geo_lock_readiness` fields before Gemma is called. Gemma receives those fields as explanatory context only. The dashboard also enriches older event files on load, so existing VisDrone outputs can show risk, alerts, and geo-lock state without rerunning the pipeline.

`garuda_edge/mission/mission_status.py` computes mission status deterministically from loaded events, tracks, summary data, mission objectives, and output root. The dashboard uses that status for mission health, scenario completion estimate, geo-lock ready tracks, and next-step support.
