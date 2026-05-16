# Edge Deployment

Phase 1 runs on a development laptop.

Future deployment path:

- YOLO -> ONNX -> TensorRT
- Gemma -> quantized local runtime
- Docker services for perception, fusion, analyst, API, and dashboard
- Jetson/edge GPU profiling for FPS, memory, thermals, and end-to-end event latency
- Persistent storage upgrade only after JSONL audit behavior is stable
