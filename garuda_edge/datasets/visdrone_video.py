from __future__ import annotations

import csv
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
LIKELY_FRAME_DIR_NAMES = {"img1", "images", "frames"}

CLASS_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}
VEHICLE_CLASS_IDS = {3, 4, 5, 8, 9}
PERSON_CLASS_IDS = {0, 1}


@dataclass
class VisDroneObject:
    frame_index: int
    target_id: int | None
    bbox_left: float
    bbox_top: float
    bbox_width: float
    bbox_height: float
    score: float | None = None
    class_id: int | None = None
    truncation: int | None = None
    occlusion: int | None = None

    @property
    def area(self) -> float:
        return max(0.0, self.bbox_width) * max(0.0, self.bbox_height)


@dataclass
class VisDroneFrameAnnotation:
    frame_index: int
    objects: list[VisDroneObject] = field(default_factory=list)


@dataclass
class VisDroneSequenceInfo:
    sequence_name: str
    sequence_dir: Path
    annotation_path: Path | None = None
    split_name: str | None = None
    sequence_style: str = "unknown"
    frame_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sequence_dir"] = str(self.sequence_dir)
        payload["annotation_path"] = str(self.annotation_path) if self.annotation_path else None
        return payload


@dataclass
class VisDroneSequenceStats:
    sequence_name: str
    sequence_dir: str
    annotation_path: str | None
    frame_count: int
    annotated_frame_count: int
    object_count_total: int
    avg_objects_per_frame: float
    avg_vehicle_count_per_frame: float
    avg_person_count_per_frame: float
    vehicle_track_count: int
    person_track_count: int
    class_counts: dict[str, int]
    class_diversity: int
    median_box_area_ratio: float
    tiny_object_ratio: float
    estimated_camera_motion_score: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def natural_key(path: Path | str) -> list[Any]:
    text = Path(path).name if isinstance(path, Path) else str(path)
    parts: list[Any] = []
    chunk = ""
    is_digit = False
    for char in text:
        if char.isdigit() != is_digit:
            if chunk:
                parts.append(int(chunk) if is_digit else chunk.lower())
            chunk = char
            is_digit = char.isdigit()
        else:
            chunk += char
    if chunk:
        parts.append(int(chunk) if is_digit else chunk.lower())
    return parts


def list_sequence_frames(sequence_dir: Path) -> list[Path]:
    sequence_dir = Path(sequence_dir)
    if not sequence_dir.exists():
        return []

    direct = [p for p in sequence_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if direct:
        return sorted(direct, key=natural_key)

    for child_name in LIKELY_FRAME_DIR_NAMES:
        child = sequence_dir / child_name
        if child.exists() and child.is_dir():
            frames = [p for p in child.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
            if frames:
                return sorted(frames, key=natural_key)

    frames = [p for p in sequence_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(frames, key=lambda p: [*natural_key(p.parent), *natural_key(p)])


def discover_visdrone_sequences(dataset_root: Path) -> list[VisDroneSequenceInfo]:
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        return []

    sequence_roots: dict[Path, list[Path]] = {}
    for directory in [dataset_root, *[p for p in dataset_root.rglob("*") if p.is_dir()]]:
        if any("det" in part.lower() for part in directory.parts):
            continue
        direct_frames = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        if not direct_frames:
            continue
        sequence_dir = directory.parent if directory.name.lower() in LIKELY_FRAME_DIR_NAMES else directory
        sequence_roots.setdefault(sequence_dir, list_sequence_frames(sequence_dir))

    sequences: list[VisDroneSequenceInfo] = []
    for sequence_dir, frames in sorted(sequence_roots.items(), key=lambda item: str(item[0])):
        if not frames:
            continue
        split_name = _find_split_name(sequence_dir)
        style = _infer_sequence_style(sequence_dir)
        annotation_path = find_annotation_for_sequence(sequence_dir, dataset_root)
        sequences.append(
            VisDroneSequenceInfo(
                sequence_name=sequence_dir.name,
                sequence_dir=sequence_dir,
                annotation_path=annotation_path,
                split_name=split_name,
                sequence_style=style,
                frame_count=len(frames),
            )
        )
    return sequences


def find_annotation_for_sequence(sequence_dir: Path, dataset_root: Path) -> Path | None:
    sequence_dir = Path(sequence_dir)
    dataset_root = Path(dataset_root)
    names = {sequence_dir.name}
    if sequence_dir.name.lower() in LIKELY_FRAME_DIR_NAMES:
        names.add(sequence_dir.parent.name)

    candidates: list[Path] = []
    for txt in dataset_root.rglob("*.txt"):
        if txt.stem in names:
            candidates.append(txt)
    if not candidates:
        return None

    def score(path: Path) -> tuple[int, int, str]:
        annotation_bonus = 0 if "annotations" in {part.lower() for part in path.parts} else 1
        shared_parts = len(set(path.parts) & set(sequence_dir.parts))
        return annotation_bonus, -shared_parts, str(path)

    return sorted(candidates, key=score)[0]


def parse_mot_annotation(annotation_path: Path) -> dict[int, VisDroneFrameAnnotation]:
    return _parse_annotation_rows(annotation_path)


def parse_vid_annotation(annotation_path: Path) -> dict[int, VisDroneFrameAnnotation]:
    return _parse_annotation_rows(annotation_path)


def parse_annotation_auto(annotation_path: Path) -> dict[int, VisDroneFrameAnnotation]:
    return _parse_annotation_rows(annotation_path)


def compute_sequence_stats(sequence_info: VisDroneSequenceInfo, sample_every_n_frames: int = 10) -> VisDroneSequenceStats:
    frames = list_sequence_frames(sequence_info.sequence_dir)
    frame_count = len(frames) or sequence_info.frame_count
    notes: list[str] = []

    annotations: dict[int, VisDroneFrameAnnotation] = {}
    if sequence_info.annotation_path and Path(sequence_info.annotation_path).exists():
        annotations = parse_annotation_auto(Path(sequence_info.annotation_path))
    else:
        notes.append("missing_annotation")

    size = load_frame_size(frames[0]) if frames else None
    if size is None:
        notes.append("frame_size_unavailable")

    class_counts: dict[str, int] = {}
    vehicle_count = 0
    person_count = 0
    vehicle_tracks: set[int] = set()
    person_tracks: set[int] = set()
    area_ratios: list[float] = []
    tiny_count = 0
    object_count_total = 0

    for frame_annotation in annotations.values():
        for obj in frame_annotation.objects:
            object_count_total += 1
            class_name = CLASS_NAMES.get(obj.class_id, "unknown")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            if obj.class_id in VEHICLE_CLASS_IDS:
                vehicle_count += 1
                if obj.target_id is not None and obj.target_id > 0:
                    vehicle_tracks.add(obj.target_id)
            if obj.class_id in PERSON_CLASS_IDS:
                person_count += 1
                if obj.target_id is not None and obj.target_id > 0:
                    person_tracks.add(obj.target_id)
            if size:
                frame_w, frame_h = size
                denom = max(1.0, float(frame_w * frame_h))
                ratio = obj.area / denom
                area_ratios.append(ratio)
                if ratio <= 0.0025:
                    tiny_count += 1

    denominator = frame_count or max(1, len(annotations))
    motion_score = _estimate_camera_motion(frames, sample_every_n_frames, notes)
    median_area = statistics.median(area_ratios) if area_ratios else 0.0

    return VisDroneSequenceStats(
        sequence_name=sequence_info.sequence_name,
        sequence_dir=str(sequence_info.sequence_dir),
        annotation_path=str(sequence_info.annotation_path) if sequence_info.annotation_path else None,
        frame_count=frame_count,
        annotated_frame_count=len(annotations),
        object_count_total=object_count_total,
        avg_objects_per_frame=round(object_count_total / denominator, 4),
        avg_vehicle_count_per_frame=round(vehicle_count / denominator, 4),
        avg_person_count_per_frame=round(person_count / denominator, 4),
        vehicle_track_count=len(vehicle_tracks),
        person_track_count=len(person_tracks),
        class_counts=dict(sorted(class_counts.items())),
        class_diversity=len([count for count in class_counts.values() if count > 0]),
        median_box_area_ratio=round(median_area, 8),
        tiny_object_ratio=round((tiny_count / len(area_ratios)) if area_ratios else 0.0, 4),
        estimated_camera_motion_score=round(motion_score, 4),
        notes=notes,
    )


def load_frame_size(frame_path: Path) -> tuple[int, int] | None:
    try:
        import cv2

        image = cv2.imread(str(frame_path))
        if image is None:
            return None
        height, width = image.shape[:2]
        return width, height
    except Exception:
        return None


def summarize_dataset(dataset_root: Path) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    sequences = discover_visdrone_sequences(dataset_root)
    annotation_files = list(dataset_root.rglob("*.txt")) if dataset_root.exists() else []
    total_frames = sum(sequence.frame_count for sequence in sequences)
    top_longest = sorted(sequences, key=lambda seq: seq.frame_count, reverse=True)[:10]
    image_dirs = _image_leaf_dirs(dataset_root) if dataset_root.exists() else []
    det_like_dirs = [path for path in image_dirs if "det" in str(path).lower()]

    return {
        "dataset_root": str(dataset_root),
        "sequence_count": len(sequences),
        "annotation_file_count": len(annotation_files),
        "total_frames": total_frames,
        "top_10_longest_sequences": [seq.to_dict() for seq in top_longest],
        "mot_style_found": any(seq.sequence_style == "mot" for seq in sequences),
        "vid_style_found": any(seq.sequence_style == "vid" for seq in sequences),
        "det_only_detected": bool(det_like_dirs and not sequences),
    }


def _parse_annotation_rows(annotation_path: Path) -> dict[int, VisDroneFrameAnnotation]:
    annotations: dict[int, VisDroneFrameAnnotation] = {}
    with Path(annotation_path).open("r", encoding="utf-8", newline="") as fp:
        for raw in fp:
            row = _split_annotation_row(raw)
            if not row:
                continue
            obj = _parse_object_row(row)
            if obj is None:
                continue
            frame_annotation = annotations.setdefault(obj.frame_index, VisDroneFrameAnnotation(frame_index=obj.frame_index))
            frame_annotation.objects.append(obj)
    return dict(sorted(annotations.items()))


def _split_annotation_row(raw: str) -> list[str]:
    stripped = raw.strip()
    if not stripped:
        return []
    if "," in stripped:
        return [part.strip() for part in next(csv.reader([stripped]))]
    return stripped.split()


def _parse_object_row(row: list[str]) -> VisDroneObject | None:
    if len(row) < 6:
        return None
    try:
        frame_index = int(float(row[0]))
        target_id = int(float(row[1])) if len(row) > 1 and row[1] != "" else None
        left, top, width, height = [float(value) for value in row[2:6]]
        score: float | None = None
        category: int | None = None
        truncation: int | None = None
        occlusion: int | None = None
        if len(row) == 7:
            category = int(float(row[6]))
        elif len(row) >= 8:
            score = float(row[6]) if row[6] != "" else None
            category = int(float(row[7])) if row[7] != "" else None
            truncation = int(float(row[8])) if len(row) >= 9 and row[8] != "" else None
            occlusion = int(float(row[9])) if len(row) >= 10 and row[9] != "" else None
    except ValueError:
        return None

    if category == 0:
        return None
    class_id = category - 1 if category is not None else None
    return VisDroneObject(
        frame_index=frame_index,
        target_id=target_id,
        bbox_left=left,
        bbox_top=top,
        bbox_width=width,
        bbox_height=height,
        score=score,
        class_id=class_id,
        truncation=truncation,
        occlusion=occlusion,
    )


def _estimate_camera_motion(frames: list[Path], sample_every_n_frames: int, notes: list[str]) -> float:
    if len(frames) < 2:
        return 0.0
    sample_step = max(1, sample_every_n_frames)
    sampled = frames[::sample_step]
    if len(sampled) < 2:
        sampled = frames[:2]

    try:
        import cv2
        import numpy as np

        scores: list[float] = []
        prev_gray = None
        for frame_path in sampled:
            image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            if prev_gray is not None:
                corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=150, qualityLevel=0.01, minDistance=8)
                if corners is not None:
                    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, image, corners, None)
                    if next_points is not None and status is not None:
                        valid = status.reshape(-1) == 1
                        if valid.any():
                            displacement = next_points[valid] - corners[valid]
                            magnitudes = np.linalg.norm(displacement.reshape(-1, 2), axis=1)
                            diagonal = max(1.0, float((image.shape[0] ** 2 + image.shape[1] ** 2) ** 0.5))
                            scores.append(float(np.median(magnitudes) / diagonal))
            prev_gray = image
        if not scores:
            notes.append("camera_motion_unavailable")
            return 0.0
        return min(1.0, float(statistics.mean(scores) * 10.0))
    except Exception:
        notes.append("camera_motion_unavailable")
        return 0.0


def _find_split_name(sequence_dir: Path) -> str | None:
    for part in sequence_dir.parts:
        if part.startswith("VisDrone2019-"):
            return part
    return None


def _infer_sequence_style(sequence_dir: Path) -> str:
    lowered = " ".join(part.lower() for part in sequence_dir.parts)
    if "mot" in lowered:
        return "mot"
    if "vid" in lowered:
        return "vid"
    return "unknown"


def _image_leaf_dirs(dataset_root: Path) -> list[Path]:
    dirs: list[Path] = []
    for directory in [dataset_root, *[p for p in dataset_root.rglob("*") if p.is_dir()]]:
        try:
            if any(p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in directory.iterdir()):
                dirs.append(directory)
        except OSError:
            continue
    return dirs
