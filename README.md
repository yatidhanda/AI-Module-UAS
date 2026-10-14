# GarudaEdge ISR Fusion Module

GarudaEdge is a serious Python MVP for a human-in-the-loop UAS ISR decision-support demo. It separates fast deterministic perception from slower contextual reasoning: YOLO detects objects, the tracker maintains object memory, telemetry is synchronized, approximate geo-tracks are estimated, deterministic confidence is scored, Gemma 4 E4B explains event packets through Ollama, and the policy engine validates every final decision for operator review.

```
video/image -> YOLO -> tracker -> telemetry sync -> geo estimate -> confidence
                                                          |
                                                          v
                                             event store / audit JSONL
                                                          |
                                                          v
                   event-driven Gemma analyst -> policy gate -> dashboard/API
```

## Safety Statement

This MVP is non-weaponized, human-in-the-loop ISR support software. "Geo-track lock" means persistent object tracking plus approximate geolocation for human review only. The policy engine blocks unsupported actions, forces `human_review_required=true`, and allows only observation, review, logging, and sensor-cueing actions.

## Setup

```bash
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

YOLO is configured to use `yolov8n.pt`. Model weights are ignored by git. Ollama should be running locally at `http://localhost:11434` with `gemma4:e4b`; if it is unavailable, the pipeline tries `gemma3:4b` and `gemma3n:e4b`, then continues with a safe fallback decision.

## Using Gemma 4 E4B Locally

```bash
brew upgrade ollama
ollama serve
ollama pull gemma4:e4b
ollama run gemma4:e4b
python scripts/run_gemma_test.py --model gemma4:e4b
```

Configured fallback models:

- `gemma3:4b`
- `gemma3n:e4b`

## Generate Telemetry

```bash
python scripts/generate_synthetic_telemetry.py --frames 1000 --fps 30 --output data/telemetry/demo_telemetry.csv
```

Telemetry is smooth and seeded: `--profile hover` (default) holds position with small drift, `--profile transect` flies straight 4 m/s legs. It is synthetic and unrelated to the video, so any drone motion in it makes parked objects appear to move on the map.

### Real flight telemetry (DJI .SRT)

Many DJI drones record a `.SRT` subtitle file next to each video with GPS and altitude for every frame (enable video captions in DJI Fly/Pilot). Pass it directly:

```bash
python scripts/run_pipeline.py --video data/input_videos/DJI_0001.MP4 --telemetry data/input_videos/DJI_0001.SRT --use-gemma false \
  --srt-gimbal-tilt-deg -60   # needed when the SRT has no gimbal angles (most consumer models)
```

Enterprise models record gimbal angles (`gb_yaw/gb_pitch/gb_roll`), which are used automatically. Otherwise the camera heading comes from the drone's direction of travel; pass `--srt-yaw-deg` for a hovering drone. Set `horizontal_fov_deg`/`vertical_fov_deg` in `configs/camera.yaml` to your camera.

### Geolocation output

Telemetry is interpolated to each frame's timestamp (angles along the shortest arc; no interpolation across gaps over 1 s). Geo uncertainty above 15 m counts as poor: it lowers confidence and blocks geo-lock (`max_geo_uncertainty_m: 15` in `configs/mission.yaml`).

Each event's `geo_estimate` is a Kalman-filtered ground position of the box's bottom-centre, with `geo_uncertainty_m` (≈1σ, never below the drone's GPS error), `speed_mps`, and `heading_deg` (only when clearly moving). The unfiltered per-frame fix is kept in `geo_raw_estimate`. Ground is assumed flat; hills and slopes add error that is not modelled.

## Add Video

Place your demo video at:

```text
data/input_videos/sampleDronefootage.mov
```

The pipeline fails with a clear error if the video path is missing.

## Run Pipeline

```bash
python scripts/run_pipeline.py --video data/input_videos/sampleDronefootage.mov --telemetry data/telemetry/demo_telemetry.csv --output data/outputs --max-frames 300 --use-gemma true
```

The pipeline refuses to write into a folder that already holds a run (runs would mix, since track IDs restart at `T-0001`). Add `--overwrite` to replace it; `operator_actions.jsonl` is always kept. Useful flags:

- `--gemma-async` runs Gemma on a background thread so slow LLM calls never stall the frame loop (for live feeds). Each event then carries the latest *finished* analysis for its track, `analyst_source_event_id` names the event it was computed from, and every analysis is logged to `analyst_decisions.jsonl`. The policy gate still validates every event.
- `yolo_device` in `configs/model.yaml` (`auto` by default) picks CUDA, then the Apple GPU (`mps`), then CPU. Docker on macOS cannot reach the Apple GPU, so run natively for GPU speed.

Outputs:

- `data/outputs/events.jsonl`
- `data/outputs/tracks.json`
- `data/outputs/summary.json`
- `data/outputs/annotated_demo.mp4`
- `data/outputs/evidence/` — one frame per new track and per confidence-level change
- `data/outputs/analyst_decisions.jsonl` (with `--gemma-async`)
- `data/outputs/track_states.jsonl` — Kalman-predicted box for every live track at `--track-output-hz` (default 1000 Hz; `0` disables). YOLO still runs once per video frame; between frames each track's constant-velocity filter extrapolates forward. These are kinematic states only: they create no events and do not go through Gemma or the policy gate.

## Dashboard

```bash
streamlit run garuda_edge/dashboard/app.py
```

## Ownship / Drone Location Layer

The dashboard home screen now shows the drone live/current location from telemetry as a read-only ownship layer. The common operating picture includes a distinct drone/ownship marker, heading line, recent trail, altitude and link-health context, and ownship-relative range/bearing values for geo-located tracks.

The current demo uses synthetic telemetry from `data/telemetry/*.csv` and writes normalized `platform_state` into new event packets. Older outputs remain supported through legacy `platform`, `telemetry`, `drone_state`, and `link` fields. Future autopilot integration can replace the synthetic source with real MAVLink, PX4, or ArduPilot telemetry.

This feature is for operator awareness and C2 situational picture only. It does not send flight commands and does not implement takeoff, land, return-to-base execution, waypoint execution, or autonomous flight control.

## Using the Operator Dashboard

The Streamlit dashboard is an operator-grade ISR review console for human-in-the-loop monitoring. Use the sidebar `Output root` selector to inspect `data/outputs` or compare VisDrone scenario folders under `data/outputs/visdrone/*`.

The mission context panel loads `configs/mission.yaml` by default and displays mission ID, phase, sector, ruleset, area of interest, task type, priority classes, and notes when present. Missing fields are shown as `Not specified`.

The top KPI cards summarize total events, active tracks, confidence levels, Gemma-analyzed events, policy corrections, operator actions, and latest frame. The map shows the drone/ownship location, heading, trail, latest geo-track locations by confidence level, and optional selected-track range line. Tabs expose mission overview, track detail, evidence, analyst/policy review, event JSON downloads, and operator action logging.

Confidence colors are intentionally sparse: high is muted green, medium is amber, low/degraded is red, and unknown/system information is gray or slate. Policy corrections and all operator actions remain visible for audit; every operational decision remains human reviewed.

The dashboard also shows mission/platform risk alerts from `configs/threat_library.yaml`. These scores are operator review cues for mission safety, platform risk, degraded link/sensor context, and object-of-interest priority. They are not hostile-intent findings and do not authorize high-consequence action. Geo-lock readiness is shown separately using mission criteria from `configs/mission.yaml`.

Gemma analyst outputs may include mission completion estimate, next-step recommendation, geo-lock readiness explanation, mission health summary, and threat/risk explanation. The deterministic mission status and policy engine remain the controlling layers.

## FastAPI

```bash
uvicorn garuda_edge.api.server:app --reload
```

Endpoints include `/health`, `/events`, `/events/latest`, `/tracks`, `/summary`, and `/operator-action`.

## Tests

```bash
python -m pytest
```

## Dataset collection and annotation (Part A)

GarudaEdge includes a separate, provenance-first dataset pipeline for permitted public
media and owned/licensed local sources. Large media, review state, CVAT staging, and
versioned YOLO datasets are stored under `GARUDA_DATA_ROOT` (`~/GarudaEdgeData` by
default), not in this checkout. Dataset dependencies are optional and are not installed
with the edge inference runtime.

```bash
source venv/bin/activate
python -m pip install -e ".[dataset]"
garuda-dataset --help
garuda-dataset stats
```

See [Part A dataset pipeline](docs/DATASET_BUILDING.md) for the exact 100–200 asset pilot,
rights quarantine, human review and CVAT workflow, strict annotation import, deterministic
build validation, disk sizing, backups, and the stable Python `data.yaml` lookup.

## Next Steps

YOLO fine-tuning: collect domain-specific labeled public/synthetic imagery, train a small detector, evaluate mAP and false positives, then export to ONNX/TensorRT for edge deployment.

Gemma LoRA fine-tuning: collect safe analyst packet examples, enforce JSON schema behavior, evaluate correction rate by the policy gate, and retain the policy engine as the final authority.

Edge roadmap: export YOLO to ONNX, convert to TensorRT, run Gemma through a quantized local runtime, split services with Docker, and profile on Jetson-class hardware.
