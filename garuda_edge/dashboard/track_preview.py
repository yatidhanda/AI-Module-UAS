from __future__ import annotations

import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_selected_track_preview(
    selected_track_id: str,
    events: list[dict[str, Any]],
    output_root: Path,
) -> dict[str, Any] | None:
    output_root = _resolve_project_path(output_root)
    track_events = [
        event for event in events if str(event.get("track_id") or event.get("event_id") or "unknown") == str(selected_track_id)
    ]
    if not track_events:
        return None

    sorted_events = sorted(track_events, key=_event_sort_key, reverse=True)
    latest = sorted_events[0]
    latest_evidence = _resolve_evidence_path(latest.get("evidence_frame_path"), output_root)
    if latest_evidence and latest_evidence.exists():
        return _preview_payload(
            image_path=latest_evidence,
            latest_event=latest,
            image_event=latest,
            source="evidence_frame",
        )

    for event in sorted_events:
        evidence_path = _resolve_evidence_path(event.get("evidence_frame_path"), output_root)
        if evidence_path and evidence_path.exists():
            return _preview_payload(
                image_path=evidence_path,
                latest_event=latest,
                image_event=event,
                source="evidence_frame",
            )

    extracted = _extract_video_frame_for_event(output_root / "annotated_demo.mp4", output_root, selected_track_id, latest)
    if extracted:
        return _preview_payload(
            image_path=extracted,
            latest_event=latest,
            image_event=latest,
            source="extracted_video_frame",
        )

    fallback = _latest_dashboard_frame(output_root)
    if fallback:
        return _preview_payload(
            image_path=fallback,
            latest_event=latest,
            image_event=latest,
            source="fallback",
        )
    return None


def render_track_highlight_image(
    image_path: str,
    bbox_xyxy: list[float] | None,
    output_root: Path,
    track_id: str,
    frame_id: int | None,
    object_class: str | None = None,
) -> str:
    output_root = _resolve_project_path(output_root)
    source_path = Path(image_path)
    if not source_path.exists():
        return str(source_path)
    output_dir = output_root / "dashboard_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_track_id = _safe_filename(track_id or "unknown")
    frame_label = "na" if frame_id is None else str(frame_id)
    output_path = output_dir / f"selected_track_{safe_track_id}_{frame_label}.jpg"

    try:
        import cv2

        image = cv2.imread(str(source_path))
        if image is None:
            return str(source_path)
        height, width = image.shape[:2]
        bbox = _valid_bbox(bbox_xyxy, width, height)
        if bbox:
            x1, y1, x2, y2 = bbox
            thickness = max(2, int(round(max(width, height) / 360)))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness)
            label = str(track_id)
            if object_class:
                label = f"{track_id} | {object_class}"
            _draw_label(cv2, image, label, x1, y1, thickness)
        cv2.imwrite(str(output_path), image)
        return str(output_path)
    except Exception:
        return str(source_path)


def _preview_payload(
    image_path: Path,
    latest_event: dict[str, Any],
    image_event: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    track_id = str(latest_event.get("track_id") or latest_event.get("event_id") or "unknown")
    image_frame_id = _int_or_none(image_event.get("frame_id"))
    latest_frame_id = _int_or_none(latest_event.get("frame_id"))
    object_class = str(latest_event.get("object_class") or image_event.get("object_class") or "unknown")
    return {
        "image_path": str(image_path),
        "event": latest_event,
        "image_event": image_event,
        "bbox_xyxy": image_event.get("bbox_xyxy") or latest_event.get("bbox_xyxy"),
        "source": source,
        "caption": f"{track_id} | {object_class} | frame {image_frame_id if image_frame_id is not None else 'n/a'} | {source}",
        "preview_frame_id": image_frame_id if image_frame_id is not None else latest_frame_id,
    }


def _extract_video_frame_for_event(
    video_path: Path,
    output_root: Path,
    track_id: str,
    event: dict[str, Any],
) -> Path | None:
    if not video_path.exists():
        return None
    frame_id = _int_or_none(event.get("frame_id"))
    output_dir = output_root / "dashboard_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"track_preview_source_{_safe_filename(track_id)}_{frame_id if frame_id is not None else 'na'}.jpg"
    if output_path.exists():
        return output_path
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target = 0 if frame_id is None else frame_id
        if frame_count > 0:
            target = max(0, min(frame_count - 1, target))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        if cv2.imwrite(str(output_path), frame):
            return output_path
    except Exception:
        return None
    return None


def _latest_dashboard_frame(output_root: Path) -> Path | None:
    frame_dir = output_root / "dashboard_frames"
    if not frame_dir.exists():
        return None
    frames = [
        path
        for path in frame_dir.glob("*.jpg")
        if path.is_file() and not path.name.startswith("selected_track_") and not path.name.startswith("track_preview_source_")
    ]
    if not frames:
        return None
    return max(frames, key=lambda path: path.stat().st_mtime)


def _resolve_evidence_path(value: Any, output_root: Path) -> Path | None:
    if not value:
        return None
    try:
        path = Path(str(value)).expanduser()
    except TypeError:
        return None
    candidates = [path] if path.is_absolute() else [output_root / path, PROJECT_ROOT / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _valid_bbox(bbox_xyxy: list[float] | None, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not isinstance(bbox_xyxy, list) or len(bbox_xyxy) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox_xyxy[:4]]
    except (TypeError, ValueError):
        return None
    x1_i = int(round(max(0, min(width - 1, x1))))
    y1_i = int(round(max(0, min(height - 1, y1))))
    x2_i = int(round(max(0, min(width - 1, x2))))
    y2_i = int(round(max(0, min(height - 1, y2))))
    if x2_i <= x1_i or y2_i <= y1_i:
        return None
    return x1_i, y1_i, x2_i, y2_i


def _draw_label(cv2: Any, image: Any, label: str, x1: int, y1: int, thickness: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(0.9, thickness * 0.22))
    text_thickness = max(1, thickness // 2)
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
    label_y1 = max(0, y1 - text_height - baseline - 8)
    label_y2 = label_y1 + text_height + baseline + 8
    label_x2 = min(image.shape[1] - 1, x1 + text_width + 12)
    cv2.rectangle(image, (x1, label_y1), (label_x2, label_y2), (80, 0, 160), -1)
    cv2.putText(image, label, (x1 + 6, label_y2 - baseline - 4), font, font_scale, (255, 255, 255), text_thickness)


def _event_sort_key(event: dict[str, Any]) -> tuple[float, float]:
    return (_float(event.get("frame_id"), default=-1.0), _float(event.get("timestamp_s"), default=-1.0))


def _float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return safe.strip("._") or "unknown"


def _resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved
