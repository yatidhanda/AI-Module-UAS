# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

GarudaEdge (`garuda_edge`) is a Python MVP for **non-weaponized, human-in-the-loop** UAS ISR decision support: YOLO detection → tracking → telemetry sync → approximate geolocation → deterministic confidence → event-driven Gemma analyst (via Ollama) → policy gate → dashboard/API. A separate, offline dataset-building package (`garuda_edge/dataset_builder`, "Part A") is not used by the runtime pipeline.

## Commands

```bash
source venv/bin/activate
pip install -r requirements.txt && pip install -e .     # runtime
python -m pip install -e ".[dataset]"                   # adds the dataset-builder extra + `garuda-dataset` CLI

python -m pytest                                        # all tests (pyproject sets testpaths=tests, pythonpath=., -q)
python -m pytest tests/test_policy.py                   # one file
python -m pytest tests/test_policy.py -k low_confidence # one test by name

# Pipeline (needs a video in data/input_videos/, which is gitignored)
python scripts/generate_synthetic_telemetry.py --frames 1000 --fps 30 --output data/telemetry/demo_telemetry.csv
python scripts/run_pipeline.py --video data/input_videos/sampleDronefootage.mov \
  --telemetry data/telemetry/demo_telemetry.csv --output data/outputs --max-frames 300 --use-gemma false

# VisDrone scenario batch (5 scenarios → data/outputs/visdrone/<scenario>/); see docs/VISDRONE_VIDEO_USAGE.md
python scripts/run_visdrone_video_batch.py --use-gemma false --max-frames 300

streamlit run garuda_edge/dashboard/app.py
uvicorn garuda_edge.api.server:app --reload
python scripts/run_gemma_test.py --model gemma4:e4b     # Ollama smoke test
```

No linter/formatter is configured. Use `--use-gemma false` for fast iteration; Gemma calls are slow and need `ollama serve` at `localhost:11434`.

## Architecture

**Orchestration lives in `scripts/run_pipeline.py`**, not in the package. It wires together the per-stage modules frame by frame and is the best place to read how the stages connect. Scripts insert the repo root into `sys.path` and resolve relative paths against the project root.

Tracking (`tracking/tracker.py`) is SORT-style: each track owns a `BoxKalmanFilter` (`tracking/kalman.py`, constant velocity, time steps in seconds), detections are matched to *predicted* boxes by Hungarian assignment (same class only), and `Track.last_bbox_xyxy` is the filtered box. `ObjectTracker.predict_tracks(t)` extrapolates without mutating state; `tracking/track_stream.py` uses it to write `track_states.jsonl` at `--track-output-hz` (default 1000) between frames, bypassing `EventStore`.

Per track, per frame, the pipeline builds one `ISREvent` (`garuda_edge/ingestion/schemas.py`, Pydantic) and enriches it in this order:
1. `geolocation.projection.project_bbox_to_ground` — pinhole ray from the box's bottom-centre (ground contact) to a flat ground plane, angle convention documented on `CameraModel`. `geo_uncertainty_m` is ~1σ, propagated numerically from GPS/altitude/attitude sigmas plus link/sync penalties. Then `geolocation.geo_tracker.GeoTrackFilter` (per-track constant-velocity Kalman in local metres, χ² outlier gate, uncertainty floored at GPS sigma) produces the filtered `geo_estimate` with `speed_mps`/`heading_deg`; the unfiltered fix is kept as `geo_raw_estimate`. Everything downstream uses the filtered estimate.
2. `confidence.scoring.score_confidence` — deterministic; folds in detection, track, GPS, link, sensor, sync, and geo-uncertainty factors; emits `level` and `degraded_flags`.
3. `fusion.FusionEngine.create_event`.
4. `threat.threat_scoring.score_threat_for_track` (driven by `configs/threat_library.yaml`) → `threat_assessment`; `mission.mission_status.evaluate_geo_lock_readiness` / `compute_mission_status` (driven by `configs/mission.yaml`). These are computed **before** Gemma and passed to it only as context.
5. `llm_analyst.GemmaAnalyst` — **event-driven, not per-frame**: called on a new track, confidence-level change, every `gemma_call_interval_frames`, or degraded conditions (`should_call_gemma`). Falls back through `ollama_fallback_models`, then to `SAFE_FALLBACK_DECISION`. With `--gemma-async`, `llm_analyst.worker.AnalystWorker` runs it on a background thread (one pending request per track, newer replaces older); events carry the latest finished decision plus `analyst_source_event_id`.
6. `policy_engine.PolicyEngine.validate` — the final authority (see Safety below). Blocked language is matched as whole words with inflections (`_BLOCKED_PATTERN`), minus a small allowlist of everyday phrases (`_BENIGN_PHRASES`, e.g. "fire hydrant"); `test_every_listed_keyword_is_blocked` keeps the pattern in step with `BLOCKED_KEYWORDS`.
7. `storage.EventStore.append_event` → `events.jsonl`; `tracks.json` / `summary.json` are kept in memory and rewritten at most once per second and on `flush()` (the pipeline's final `summary.json` then replaces the store's). The pipeline refuses an output dir that already has `events.jsonl` unless `--overwrite` (which never deletes `operator_actions.jsonl`).

Telemetry is looked up through `ingestion.telemetry.TelemetryTimeline` (bisect, interpolated to the frame time). `HIGH_GEO_UNCERTAINTY_M` (15 m, in `geolocation/projection.py`) is the shared "poor geo" threshold for confidence scoring and the dashboard; `mission.yaml` and `threat_library.yaml` use the same value.

**Dashboard (`garuda_edge/dashboard/`)** reads output folders, not the live pipeline. `data_loader.py` is the compatibility layer: it handles older event files (legacy `platform`/`telemetry`/`drone_state`/`link` fields vs. newer `platform_state`) and re-scores `threat_assessment` / `geo_lock_readiness` at load time, so schema additions must keep older outputs loadable. `ownship.py` derives the read-only drone position/trail layer. Output-root discovery covers `data/outputs` and `data/outputs/visdrone/*`.

**API (`garuda_edge/api/server.py`)** is a thin FastAPI wrapper over `EventStore` with a hard-coded, CWD-relative `data/outputs`.

**Telemetry** loads via `ingestion.telemetry.load_telemetry`: GarudaEdge CSV, or DJI flight-subtitle `.SRT` (`ingestion/dji_srt.py`, tolerant of consumer/Enterprise/legacy layouts; missing gimbal tilt/yaw come from `--srt-gimbal-tilt-deg`/`--srt-yaw-deg`). Synthetic CSVs come from `ingestion/synthetic_telemetry.py` (smooth OU processes, `hover`/`transect` profiles, seeded); they are unrelated to any video, so drone motion in them moves stationary objects on the map.

**Configs (`configs/*.yaml`)** drive model/Ollama settings (`model.yaml`), camera intrinsics (`camera.yaml`), mission objectives and geo-lock criteria (`mission.yaml`), risk scoring (`threat_library.yaml`), and the dataset catalog/taxonomy (`datasets.yaml`, `taxonomy/garuda_objects_v1.yaml`).

### Dataset builder (Part A)

`garuda-dataset` CLI (`dataset_builder/cli.py`): discover → download → verify → deduplicate → extract-frames → review-report/review → export-for-annotation (CVAT) → import-annotations → build-dataset → validate-dataset. Key invariants (details in `docs/DATASET_BUILDING.md`):
- All large data lives under `GARUDA_DATA_ROOT` (default `~/GarudaEdgeData`), never in the checkout; metadata is in SQLite there.
- Provenance and rights are first-class: DVIDS and search-engine (Brave/SerpAPI) results are leads with `permission_decision=pending` until a human approves; pending rights cannot enter a build.
- Annotation export writes **zero label files**; import is all-or-nothing and rejects empty labels except on reviewed negatives. Automated steps mark state and never delete originals or create boxes/classes.
- Built versions are immutable; splits are deterministic and grouped by source group / near-duplicate cluster.
- Network providers use pacing, retries, size limits, and a fail-closed robots check; tests use JSON fixtures in `tests/fixtures/` rather than live HTTP.

## Safety constraints (do not weaken)

- `PolicyEngine` validates every analyst decision: decisions/actions must be in `ALLOWED_ACTIONS` (`llm_analyst/prompts.py`), blocked keywords (`engage`, `fire`, `strike`, `weapon`, …) trigger replacement, `human_review_required` is forced `true`, and low confidence forces `mark_low_confidence`. Corrections are recorded in `audit_notes`.
- "Threat" scores mean mission/platform risk for human review, not hostile intent. "Geo-track lock" means persistent tracking + approximate geolocation for review only.
- The ownship layer is read-only telemetry display; nothing in the codebase sends flight commands.
- Deterministic layers (confidence, mission status, policy) control outcomes; Gemma only explains.
