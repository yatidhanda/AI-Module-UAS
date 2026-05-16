# Demo Guide

1. Activate the environment.

```bash
source venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Generate synthetic telemetry.

```bash
python scripts/generate_synthetic_telemetry.py --frames 1000 --fps 30
```

4. Place a demo video at `data/input_videos/demo.mp4`.

5. Run the fusion pipeline.

```bash
python scripts/run_pipeline.py --video data/input_videos/demo.mp4 --telemetry data/telemetry/demo_telemetry.csv --output data/outputs --max-frames 300 --use-gemma true
```

6. Open the operator dashboard.

```bash
streamlit run garuda_edge/dashboard/app.py
```

7. Start the API in another terminal.

```bash
uvicorn garuda_edge.api.server:app --reload
```
