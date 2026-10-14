# GarudaEdge ISR Fusion Module — CPU runtime image (pipeline, dashboard, API).
#
#   docker build -t garuda-edge .
#   docker run --rm -p 8501:8501 -v "$PWD/data:/app/data" garuda-edge                     # Streamlit dashboard
#   docker run --rm -p 8000:8000 -v "$PWD/data:/app/data" garuda-edge \
#       uvicorn garuda_edge.api.server:app --host 0.0.0.0 --port 8000                       # FastAPI
#   docker run --rm -v "$PWD/data:/app/data" garuda-edge \
#       python scripts/run_pipeline.py --video data/input_videos/sampleDronefootage.mov \
#       --telemetry data/telemetry/demo_telemetry.csv --output data/outputs --use-gemma false
#   docker run --rm garuda-edge python -m pytest                                            # tests
#
# Ollama is not bundled. configs/model.yaml points at localhost:11434, which inside a
# container is the container itself; to reach Ollama on the host, mount an edited config
# with ollama_base_url: http://host.docker.internal:11434 (-v ./model.yaml:/app/configs/model.yaml).
# Without Ollama the pipeline uses its safe fallback decision.

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

# Shared libraries required by opencv-python (pulled in by requirements and ultralytics).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first so ultralytics doesn't pull the multi-GB CUDA wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Editable install: dashboard and Ollama client resolve configs/ relative to the source tree.
RUN pip install --no-deps -e . \
    && python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" \
    && useradd --create-home --uid 1000 garuda \
    && chown -R garuda:garuda /app

USER garuda

EXPOSE 8501 8000

CMD ["streamlit", "run", "garuda_edge/dashboard/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
