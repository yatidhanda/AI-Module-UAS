# VisDrone Video Usage

GarudaEdge supports VisDrone MOT/VID sequence folders for real drone video tracking tests. This workflow is for non-weaponized, human-in-the-loop ISR/search/monitoring evaluation only: detection, tracking, approximate geolocation, confidence scoring, summarization, sensor cueing, operator notification, logging, and review.

## DET vs MOT/VID

VisDrone DET folders are still-image detection datasets. They are useful for detector evaluation, but they do not stress temporal tracking, reacquisition, camera motion, or confidence changes across time.

VisDrone MOT and VID folders contain ordered image sequences and usually have per-sequence annotation files. They are better for GarudaEdge tracking tests because the pipeline can turn a sequence into an MP4, run YOLO/tracking/geo/confidence over time, and compare scenario outputs.

## Expected Folder Structure

Place the dataset under:

```text
data/datasets/VisDrone/
```

Supported layouts include MOT-style folders:

```text
data/datasets/VisDrone/
  VisDrone2019-MOT-train/
    sequences/
      uav0000013_00000_v/
        0000001.jpg
        0000002.jpg
    annotations/
      uav0000013_00000_v.txt
```

VID-style mirrors are also supported:

```text
data/datasets/VisDrone/
  VisDrone2019-VID-train/
    sequences/
      sequence_name/
        img1/
          000001.jpg
    annotations/
      sequence_name.txt
```

The discovery code searches recursively because Kaggle mirrors often add extra nesting.

## Validate The Dataset

```bash
python scripts/prepare_visdrone_sequences.py --dataset-root data/datasets/VisDrone
```

This prints discovered sequence count, annotation file count, total frames, MOT/VID detection, and the top 10 longest sequences. If only DET image folders are present, it prints:

```text
Only VisDrone DET image folders detected. For video tracking tests, download VisDrone MOT/VID sequence data or use existing drone videos.
```

## Curate Scenario Sequences

```bash
python scripts/curate_visdrone_video_scenarios.py --dataset-root data/datasets/VisDrone --limit 10
```

The curation step scores sequences for:

- `vehicle_heavy_road`
- `urban_intersection`
- `pedestrian_overhead`
- `high_altitude_tiny_vehicle`
- `moving_drifting_camera`

It writes JSON and CSV manifests to:

```text
data/sample_events/visdrone_video_manifests/
```

## Build MP4 Scenario Videos

```bash
python scripts/build_visdrone_scenario_videos.py --manifest-dir data/sample_events/visdrone_video_manifests --output-dir data/input_videos/visdrone --fps 15 --max-frames 300
```

The builder picks the top-ranked sequence per manifest and writes:

```text
data/input_videos/visdrone/vehicle_heavy_road.mp4
data/input_videos/visdrone/urban_intersection.mp4
data/input_videos/visdrone/pedestrian_overhead.mp4
data/input_videos/visdrone/high_altitude_tiny_vehicle.mp4
data/input_videos/visdrone/moving_drifting_camera.mp4
data/input_videos/visdrone/manifest_summary.json
```

## Run Scenario Batch Tests

Run fast YOLO/tracking/confidence tests without Gemma:

```bash
python scripts/run_visdrone_video_batch.py --use-gemma false --max-frames 300
```

Run a shorter Gemma-enabled batch:

```bash
python scripts/run_visdrone_video_batch.py --use-gemma true --max-frames 60
```

Gemma remains event-throttled by the existing pipeline. It is not called every frame.

Each scenario writes a separate output folder:

```text
data/outputs/visdrone/vehicle_heavy_road/
data/outputs/visdrone/urban_intersection/
data/outputs/visdrone/pedestrian_overhead/
data/outputs/visdrone/high_altitude_tiny_vehicle/
data/outputs/visdrone/moving_drifting_camera/
```

## Open The Dashboard

```bash
streamlit run garuda_edge/dashboard/app.py
```

Use the sidebar `Output root` selector to compare `data/outputs` with any existing VisDrone scenario output folder.

## Interpreting Results

Use detection count to see whether YOLO is finding enough candidate objects for a scenario.

Use track stability and confidence levels to compare tracking performance across dense traffic, pedestrian-heavy overhead scenes, tiny-object scenes, and moving-camera scenes.

Use Gemma summary quality only when `--use-gemma true`; summaries should describe uncertainty using supplied event data and must remain human-review oriented.

Use policy corrections to identify model outputs that needed safety or schema cleanup.

Use operator actions to record human acknowledgments, recheck requests, or false-positive markings beside each scenario output folder.
