from __future__ import annotations

try:
    from loguru import logger
except Exception:
    import logging

    logger = logging.getLogger(__name__)

from garuda_edge.ingestion.schemas import Detection


def resolve_device(device: str | None) -> str:
    """Map "auto" (or nothing) to the fastest available torch backend: CUDA, Apple MPS, then CPU."""
    if device and device.lower() != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class YOLODetector:
    """Thin adapter around Ultralytics YOLO for the fast visual loop."""

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.25, device: str | None = None):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = resolve_device(device)
        self.model = None
        self.names: dict[int, str] = {}
        try:
            from ultralytics import YOLO

            self.model = YOLO(model_path)
            self.names = {int(k): str(v) for k, v in getattr(self.model, "names", {}).items()}
            logger.info(f"YOLO model {model_path} on device {self.device}")
        except Exception as exc:
            logger.warning(f"YOLO model unavailable: {exc}")

    def predict(self, frame, frame_id: int, timestamp_s: float) -> list[Detection]:
        if self.model is None:
            return []
        try:
            kwargs = {"conf": self.conf_threshold, "verbose": False, "device": self.device}
            results = self.model.predict(frame, **kwargs)
        except Exception as exc:
            logger.warning(f"YOLO prediction failed on frame {frame_id}: {exc}")
            return []

        detections: list[Detection] = []
        if not results:
            return detections
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return detections
        for box in boxes:
            try:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                bbox = [float(v) for v in box.xyxy[0].tolist()]
                detections.append(
                    Detection(
                        frame_id=frame_id,
                        timestamp_s=timestamp_s,
                        class_id=class_id,
                        class_name=self.names.get(class_id, f"class_{class_id}"),
                        confidence=confidence,
                        bbox_xyxy=bbox,
                    )
                )
            except Exception as exc:
                logger.warning(f"Skipping malformed YOLO detection: {exc}")
        return detections
