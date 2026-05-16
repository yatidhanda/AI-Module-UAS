from __future__ import annotations

from pathlib import Path

from garuda_edge.datasets.visdrone_video import list_sequence_frames

try:
    from loguru import logger
except Exception:
    import logging

    logger = logging.getLogger(__name__)


def count_frames(sequence_dir: Path) -> int:
    return len(list_sequence_frames(Path(sequence_dir)))


def extract_sequence_clip(sequence_dir: Path, start_frame: int, max_frames: int) -> list[Path]:
    frames = list_sequence_frames(Path(sequence_dir))
    start = max(0, int(start_frame))
    end = start + max(0, int(max_frames))
    return frames[start:end]


def build_mp4_from_sequence(
    sequence_dir: Path,
    output_path: Path,
    fps: int | float = 15,
    max_frames: int = 300,
    resize_width: int | None = None,
) -> Path:
    import cv2

    sequence_dir = Path(sequence_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = extract_sequence_clip(sequence_dir, start_frame=0, max_frames=max_frames)
    if not frames:
        raise ValueError(f"No image frames found in sequence directory: {sequence_dir}")

    writer = None
    written = 0
    try:
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                logger.warning(f"Skipping unreadable frame: {frame_path}")
                continue
            if resize_width:
                frame = _resize_to_width(frame, resize_width)
            height, width = frame.shape[:2]
            if writer is None:
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(fps),
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer for {output_path}")
                logger.info(f"Writing {output_path} from {sequence_dir} at {width}x{height}, {fps} fps")
            writer.write(frame)
            written += 1
    finally:
        if writer is not None:
            writer.release()

    if written == 0:
        raise ValueError(f"No readable image frames found in sequence directory: {sequence_dir}")
    logger.info(f"Wrote {written} frames to {output_path}")
    return output_path


def _resize_to_width(frame, resize_width: int):
    import cv2

    height, width = frame.shape[:2]
    if width == resize_width:
        return frame
    scale = resize_width / float(width)
    resize_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
