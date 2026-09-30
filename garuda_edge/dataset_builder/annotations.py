from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import yaml

from .config import DatasetSettings
from .collection import write_workspace_manifests
from .media import sha256_file
from .models import BoundingBox
from .store import MetadataStore
from .taxonomy import Taxonomy


LABEL_GUIDE = """# GarudaEdge bounding-box label guide

This staging export is **UNANNOTATED**. Candidate categories and web captions are review
cues only; they are not labels. Do not treat an absent label as a verified negative.

1. Draw a tight visible bounding box around every relevant visible object in the taxonomy.
2. Label what is visually supportable. Do not infer exact platform identity, affiliation,
   intent, or payload from a caption.
3. For occluded objects, box the visible extent consistently. For truncated objects, clip
   the box to the image boundary. Skip objects that cannot be localized reliably.
4. When the class itself is ambiguous, leave the image unresolved and return it for review;
   do not force `unknown` or `needs_review` into a training class.
5. A negative image must be explicitly marked `negative` in GarudaEdge review before import.
   Only those reviewed negatives may have a deliberately empty label file.
6. Every relevant visible object must be annotated, including civilian lookalikes and small
   objects that remain identifiable. Do not annotate reflections, drawings, or screen images
   unless that is intentionally part of the dataset design and documented by the reviewer.
"""


def export_for_annotation(
    settings: DatasetSettings,
    taxonomy: Taxonomy,
    *,
    name: str,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    store = MetadataStore(settings.database_path)
    rows = store.list_assets(
        where=(
            "permission_decision='approved' AND review_status IN ('accepted','negative') "
            "AND annotation_status='unannotated' AND duplicate_of IS NULL "
            "AND media_type='image' AND download_status IN ('downloaded','local') "
            "AND local_path IS NOT NULL AND sha256 IS NOT NULL"
        ),
        limit=limit,
        order_by="source_group_id, asset_id",
    )
    root = settings.annotation_root / "staging" / name
    report = {"name": name, "assets": len(rows), "path": str(root), "dry_run": dry_run, "status": "UNANNOTATED"}
    if dry_run:
        report["asset_ids"] = [row["asset_id"] for row in rows]
        return report
    if root.exists():
        raise FileExistsError(f"Annotation staging export already exists: {root}")
    images_root = root / "images"
    images_root.mkdir(parents=True)
    manifest_path = root / "manifest.jsonl"
    train_lines = []
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for row in rows:
            source = Path(str(row["local_path"]))
            suffix = source.suffix.lower()
            target_name = f"{row['asset_id']}{suffix}"
            target = images_root / target_name
            shutil.copy2(source, target)
            train_lines.append(f"images/{target_name}")
            manifest.write(
                json.dumps(
                    {
                        "asset_id": row["asset_id"],
                        "file": f"images/{target_name}",
                        "sha256": row["sha256"],
                        "width": row["width"],
                        "height": row["height"],
                        "source_group_id": row["source_group_id"],
                        "candidate_category": row["candidate_category"],
                        "review_status": row["review_status"],
                        "verified_negative": row["review_status"] == "negative",
                        "annotation_status": "UNANNOTATED",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    (root / "obj.names").write_text("\n".join(taxonomy.names) + "\n", encoding="utf-8")
    (root / "obj.data").write_text(
        f"classes = {len(taxonomy.names)}\ntrain = train.txt\nnames = obj.names\nbackup = backup/\n",
        encoding="utf-8",
    )
    (root / "train.txt").write_text("\n".join(train_lines) + ("\n" if train_lines else ""), encoding="utf-8")
    (root / "README_UNANNOTATED.md").write_text(LABEL_GUIDE, encoding="utf-8")
    (root / "taxonomy.yaml").write_text(taxonomy.path.read_text(encoding="utf-8"), encoding="utf-8")
    zip_path = root.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root))
    report["zip_path"] = str(zip_path)
    report["label_files_created"] = 0
    return report


def import_annotations(
    settings: DatasetSettings,
    taxonomy: Taxonomy,
    *,
    source: str | Path,
    annotation_version: str,
    reviewer: str | None,
    dry_run: bool = False,
) -> dict[str, Any]:
    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Annotation export does not exist: {source_path}")
    settings.reports_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="garuda-cvat-") as temp_dir:
        root = Path(temp_dir)
        if source_path.is_file():
            _safe_extract_zip(source_path, root)
        else:
            root = source_path
        result = _validate_annotation_tree(
            settings,
            taxonomy,
            root,
            source_path=source_path,
            annotation_version=annotation_version,
            reviewer=reviewer,
        )
    report_path = settings.reports_root / f"annotation_import_{_safe_name(annotation_version)}.json"
    result["dry_run"] = dry_run
    result["report_path"] = str(report_path)
    if result["errors"]:
        result["success"] = False
        report_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result
    if not dry_run:
        store = MetadataStore(settings.database_path)
        for item in result.pop("validated_items"):
            store.set_annotation(
                asset_id=item["asset_id"],
                taxonomy_name=taxonomy.name,
                taxonomy_version=taxonomy.version,
                boxes=item["boxes"],
                annotation_version=item["annotation_version"],
                reviewer=item["reviewer"],
                source_export=str(source_path),
                metadata=item["metadata"],
            )
            store.record_event(
                "annotations_imported",
                asset_id=item["asset_id"],
                details={"annotation_version": item["annotation_version"], "reviewer": item["reviewer"], "boxes": len(item["boxes"])},
            )
        write_workspace_manifests(settings, store)
    else:
        result.pop("validated_items")
    result["success"] = True
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _validate_annotation_tree(
    settings: DatasetSettings,
    taxonomy: Taxonomy,
    root: Path,
    *,
    source_path: Path,
    annotation_version: str,
    reviewer: str | None,
) -> dict[str, Any]:
    errors: list[str] = []
    found_names = _read_export_names(root)
    if found_names is None:
        errors.append("Missing obj.names or data.yaml taxonomy in annotation export")
    elif found_names != taxonomy.names:
        errors.append(f"Taxonomy mismatch: expected {taxonomy.names!r}, export contains {found_names!r}")
    annotation_metadata = _read_annotation_metadata(root)
    image_paths = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
    ]
    found_stems = {path.stem for path in image_paths}
    listed_stems: set[str] = set()
    for list_name in ("train.txt", "val.txt", "test.txt"):
        for list_path in root.rglob(list_name):
            for line in list_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    listed_stems.add(Path(line.strip()).stem)
    for missing in sorted(listed_stems - found_stems):
        errors.append(f"{missing}: image listed by the CVAT export is missing")
    if len(found_stems) != len(image_paths):
        errors.append("Duplicate stable image identities appear in the annotation export")
    store = MetadataStore(settings.database_path)
    validated: list[dict[str, Any]] = []
    for image_path in sorted(image_paths):
        asset_id = image_path.stem
        asset = store.get_asset(asset_id)
        if not asset:
            errors.append(f"Unknown or unstable image identity: {image_path.name}")
            continue
        if asset["review_status"] not in {"accepted", "negative"}:
            errors.append(f"{asset_id}: review_status={asset['review_status']!r}; expected accepted or negative")
            continue
        if asset["permission_decision"] != "approved":
            errors.append(f"{asset_id}: rights are not approved")
            continue
        if sha256_file(image_path) != asset["sha256"]:
            errors.append(f"{asset_id}: image SHA-256 differs from the staged source")
            continue
        dimensions = _image_dimensions(image_path)
        if dimensions != (int(asset["width"]), int(asset["height"])):
            errors.append(f"{asset_id}: dimensions {dimensions} do not match metadata {(asset['width'], asset['height'])}")
            continue
        label_path = _find_label(root, asset_id)
        if label_path is None:
            errors.append(f"{asset_id}: missing reviewed YOLO label file")
            continue
        boxes, label_errors = _parse_label(label_path, taxonomy.ids, int(asset["width"]), int(asset["height"]))
        errors.extend(f"{asset_id}: {message}" for message in label_errors)
        if label_errors:
            continue
        if asset["review_status"] == "accepted" and not boxes:
            errors.append(f"{asset_id}: accepted positive has an empty label file; mark it negative explicitly or annotate it")
            continue
        if asset["review_status"] == "negative" and boxes:
            errors.append(f"{asset_id}: reviewed negative contains {len(boxes)} boxes")
            continue
        per_item = annotation_metadata.get(asset_id, {})
        validated.append(
            {
                "asset_id": asset_id,
                "boxes": [box.to_dict() for box in boxes],
                "annotation_version": str(per_item.get("annotation_version") or annotation_version),
                "reviewer": per_item.get("reviewer") or reviewer,
                "metadata": {
                    "label_file": str(label_path.relative_to(root)),
                    "source_export": str(source_path),
                    **{k: v for k, v in per_item.items() if k not in {"annotation_version", "reviewer"}},
                },
            }
        )
    if not image_paths:
        errors.append("No images found in CVAT export")
    return {
        "source": str(source_path),
        "annotation_version": annotation_version,
        "reviewer": reviewer,
        "images_found": len(image_paths),
        "validated": len(validated),
        "boxes": sum(len(item["boxes"]) for item in validated),
        "errors": errors,
        "validated_items": validated,
    }


def _read_export_names(root: Path) -> list[str] | None:
    candidates = sorted(root.rglob("obj.names"))
    if candidates:
        return [line.strip() for line in candidates[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    for path in [*root.rglob("data.yaml"), *root.rglob("dataset.yaml")]:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        names = payload.get("names")
        if isinstance(names, list):
            return [str(item) for item in names]
        if isinstance(names, dict):
            return [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
    return None


def _read_annotation_metadata(root: Path) -> dict[str, dict[str, Any]]:
    candidates = sorted(root.rglob("annotation_metadata.json"))
    if not candidates:
        return {}
    payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    items = payload.get("items", payload) if isinstance(payload, dict) else {}
    return {str(key): dict(value) for key, value in items.items() if isinstance(value, dict)}


def _find_label(root: Path, asset_id: str) -> Path | None:
    candidates = [path for path in root.rglob(f"{asset_id}.txt") if path.name not in {"train.txt", "val.txt", "test.txt"}]
    return sorted(candidates)[0] if candidates else None


def _parse_label(path: Path, valid_ids: set[int], image_width: int, image_height: int) -> tuple[list[BoundingBox], list[str]]:
    boxes: list[BoundingBox] = []
    errors: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"line {line_number}: expected 5 fields, found {len(fields)}")
            continue
        try:
            class_id = int(fields[0])
            x, y, width, height = (float(value) for value in fields[1:])
        except ValueError:
            errors.append(f"line {line_number}: non-numeric YOLO coordinate")
            continue
        values = (x, y, width, height)
        if class_id not in valid_ids:
            errors.append(f"line {line_number}: invalid class ID {class_id}")
        if not all(value == value and abs(value) != float("inf") for value in values):
            errors.append(f"line {line_number}: coordinate is not finite")
        elif width <= 0 or height <= 0:
            errors.append(f"line {line_number}: zero or negative box area")
        elif not all(0 <= value <= 1 for value in values):
            errors.append(f"line {line_number}: normalized coordinate outside [0,1]")
        elif x - width / 2 < -1e-8 or y - height / 2 < -1e-8 or x + width / 2 > 1 + 1e-8 or y + height / 2 > 1 + 1e-8:
            errors.append(f"line {line_number}: box extends outside image bounds")
        elif width * image_width < 1e-6 or height * image_height < 1e-6:
            errors.append(f"line {line_number}: zero-area box in pixel coordinates")
        else:
            boxes.append(BoundingBox(class_id, x, y, width, height))
    return boxes, errors


def _image_dimensions(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Annotation import requires Pillow") from exc
    with Image.open(path) as image:
        return image.size


def _safe_extract_zip(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(source) as archive:
        root = destination.resolve()
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe path in annotation ZIP: {member.filename}")
        archive.extractall(destination)


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
