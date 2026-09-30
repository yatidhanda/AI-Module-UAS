from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import DatasetSettings
from .media import sha256_file
from .store import MetadataStore
from .taxonomy import Taxonomy


SPLITS = ("train", "val", "test")


class DatasetBuildError(RuntimeError):
    pass


def build_dataset(
    settings: DatasetSettings,
    taxonomy: Taxonomy,
    *,
    version: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", version):
        raise ValueError("Version must contain only letters, digits, dot, underscore, and hyphen")
    settings.ensure_layout()
    store = MetadataStore(settings.database_path)
    eligible = store.list_assets(
        where=(
            "media_type='image' AND download_status IN ('downloaded','local') "
            "AND permission_decision='approved' AND review_status IN ('accepted','negative') "
            "AND annotation_status='validated' AND duplicate_of IS NULL "
            "AND local_path IS NOT NULL AND sha256 IS NOT NULL AND perceptual_hash IS NOT NULL"
        ),
        order_by="source_group_id, asset_id",
    )
    class_counts: Counter[int] = Counter()
    prepared: list[dict[str, Any]] = []
    preflight_errors: list[str] = []
    for asset in eligible:
        if not store.has_event(str(asset["asset_id"]), "dedup_examined"):
            preflight_errors.append(f"{asset['asset_id']}: deduplication has not been run")
        annotation = store.get_annotation(str(asset["asset_id"]))
        if not annotation:
            preflight_errors.append(f"{asset['asset_id']}: missing annotation record")
            continue
        boxes = annotation["boxes"]
        if asset["review_status"] == "accepted" and not boxes:
            preflight_errors.append(f"{asset['asset_id']}: accepted positive has no boxes")
        if asset["review_status"] == "negative" and boxes:
            preflight_errors.append(f"{asset['asset_id']}: reviewed negative has boxes")
        for box in boxes:
            class_counts[int(box["class_id"])] += 1
        prepared.append({"asset": asset, "annotation": annotation})
    validation_cfg = settings.dataset_config.get("validation") or {}
    minimum = int(validation_cfg.get("minimum_class_instances", 1))
    for class_id, class_name in enumerate(taxonomy.names):
        if class_counts[class_id] < minimum:
            preflight_errors.append(
                f"class {class_id} ({class_name}) has {class_counts[class_id]} instances; minimum is {minimum}"
            )
    if not prepared:
        preflight_errors.append("no training-ready assets")
    if preflight_errors:
        raise DatasetBuildError("Dataset preflight failed:\n- " + "\n- ".join(preflight_errors))

    split_cfg = settings.dataset_config.get("split") or {}
    proportions = {split: float(split_cfg.get(split, default)) for split, default in zip(SPLITS, (0.7, 0.2, 0.1))}
    if abs(sum(proportions.values()) - 1.0) > 1e-8 or any(value <= 0 for value in proportions.values()):
        raise ValueError(f"Split proportions must be positive and sum to 1.0: {proportions}")
    seed = int(split_cfg.get("seed", 20260505))
    assignments = _assign_splits(prepared, proportions, seed)
    planned_counts = Counter(assignments.values())
    if dry_run:
        return {
            "dry_run": True,
            "eligible_assets": len(prepared),
            "boxes_per_class": {taxonomy.names[key]: value for key, value in sorted(class_counts.items())},
            "splits": dict(planned_counts),
            "target": str(settings.builds_root / version),
        }

    target = settings.builds_root / version
    if target.exists():
        raise FileExistsError(f"Versioned dataset already exists and will not be overwritten: {target}")
    settings.builds_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{version}-", dir=settings.builds_root))
    try:
        result = _write_dataset(
            temporary,
            settings,
            taxonomy,
            version,
            prepared,
            assignments,
            proportions,
            seed,
            store,
        )
        validation = validate_dataset(temporary, settings=settings, taxonomy=taxonomy, write_report=True)
        if not validation["valid"]:
            raise DatasetBuildError("Generated dataset failed validation:\n- " + "\n- ".join(validation["errors"]))
        os.replace(temporary, target)
        # data.yaml contains an absolute temporary path while building; make it point to
        # the immutable final version and validate once more.
        _write_data_yaml(target, taxonomy, target)
        final_validation = validate_dataset(target, settings=settings, taxonomy=taxonomy, write_report=True)
        if not final_validation["valid"]:
            raise DatasetBuildError("Final dataset failed validation:\n- " + "\n- ".join(final_validation["errors"]))
        state = {
            "dataset_name": settings.dataset_name,
            "version": version,
            "data_yaml": str((target / "data.yaml").resolve()),
            "selected_at": datetime.now(timezone.utc).isoformat(),
        }
        settings.state_root.mkdir(parents=True, exist_ok=True)
        (settings.state_root / "selected_dataset.yaml").write_text(
            yaml.safe_dump(state, sort_keys=False), encoding="utf-8"
        )
        result.update(
            {
                "path": str(target.resolve()),
                "data_yaml": str((target / "data.yaml").resolve()),
                "validation_report": str((target / "validation_report.json").resolve()),
                "valid": True,
            }
        )
        return result
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _assign_splits(
    prepared: list[dict[str, Any]], proportions: dict[str, float], seed: int
) -> dict[str, str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for item in prepared:
        asset = item["asset"]
        # Source groups are the primary split unit. A near-duplicate cluster is included
        # in the key as a second guard; non-keeper duplicates are excluded from builds.
        group = str(asset["source_group_id"])
        if asset.get("near_duplicate_cluster"):
            group += f"|{asset['near_duplicate_cluster']}"
        groups[group].append(str(asset["asset_id"]))
    ordered = sorted(
        groups.items(),
        key=lambda item: hashlib.sha256(f"{seed}:{item[0]}".encode("utf-8")).hexdigest(),
    )
    total = sum(len(items) for _, items in ordered)
    target = {split: proportions[split] * total for split in SPLITS}
    counts = {split: 0 for split in SPLITS}
    group_split: dict[str, str] = {}
    # If there are enough independent groups, deliberately seed every split so a
    # deceptively train-only dataset cannot pass validation.
    start = 0
    if len(ordered) >= len(SPLITS):
        permutation = sorted(
            SPLITS,
            key=lambda split: hashlib.sha256(f"{seed}:split:{split}".encode()).hexdigest(),
        )
        for (group, items), split in zip(ordered[: len(SPLITS)], permutation):
            group_split[group] = split
            counts[split] += len(items)
        start = len(SPLITS)
    for group, items in ordered[start:]:
        deficits = {split: target[split] - counts[split] for split in SPLITS}
        split = max(
            SPLITS,
            key=lambda name: (
                deficits[name],
                hashlib.sha256(f"{seed}:{group}:{name}".encode()).hexdigest(),
            ),
        )
        group_split[group] = split
        counts[split] += len(items)
    assignments: dict[str, str] = {}
    for group, asset_ids in groups.items():
        for asset_id in asset_ids:
            assignments[asset_id] = group_split[group]
    return assignments


def _write_dataset(
    root: Path,
    settings: DatasetSettings,
    taxonomy: Taxonomy,
    version: str,
    prepared: list[dict[str, Any]],
    assignments: dict[str, str],
    proportions: dict[str, float],
    seed: int,
    store: MetadataStore,
) -> dict[str, Any]:
    for split in SPLITS:
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    manifest_rows: list[dict[str, Any]] = []
    licenses: dict[tuple[str, str, str], dict[str, str]] = {}
    class_boxes: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_viewpoint: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    for item in sorted(prepared, key=lambda value: str(value["asset"]["asset_id"])):
        asset, annotation = item["asset"], item["annotation"]
        split = assignments[str(asset["asset_id"])]
        source = Path(str(asset["local_path"]))
        extension = source.suffix.lower()
        filename = f"{asset['asset_id']}{extension}"
        image_target = root / "images" / split / filename
        _link_or_copy(source, image_target)
        label_target = root / "labels" / split / f"{asset['asset_id']}.txt"
        label_target.write_text(
            "\n".join(
                f"{int(box['class_id'])} {float(box['x_center']):.8f} {float(box['y_center']):.8f} {float(box['width']):.8f} {float(box['height']):.8f}"
                for box in annotation["boxes"]
            ) + ("\n" if annotation["boxes"] else ""),
            encoding="utf-8",
        )
        for box in annotation["boxes"]:
            class_boxes[taxonomy.names[int(box["class_id"])]] += 1
        by_source[str(asset["provider"])] += 1
        by_viewpoint[str(asset.get("viewpoint") or "unspecified")] += 1
        by_split[split] += 1
        manifest_rows.append(
            {
                "asset_id": asset["asset_id"],
                "file": f"images/{split}/{filename}",
                "label_file": f"labels/{split}/{asset['asset_id']}.txt",
                "split": split,
                "sha256": asset["sha256"],
                "width": asset["width"],
                "height": asset["height"],
                "provider": asset["provider"],
                "provider_id": asset["provider_id"],
                "source_page_url": asset["source_page_url"],
                "media_url": asset["media_url"],
                "retrieval_timestamp": asset["retrieval_timestamp"],
                "creator_credit": asset["creator_credit"],
                "rights_statement": asset["rights_statement"],
                "rights_url": asset["rights_url"],
                "permission_decision": asset["permission_decision"],
                "original_search_term": asset["original_search_term"],
                "candidate_category": asset["candidate_category"],
                "review_status": asset["review_status"],
                "annotation_status": asset["annotation_status"],
                "annotation_version": annotation["annotation_version"],
                "reviewer": annotation["reviewer"] or asset["reviewer"],
                "source_group_id": asset["source_group_id"],
                "near_duplicate_cluster": asset["near_duplicate_cluster"],
                "parent_asset_id": asset["parent_asset_id"],
                "frame_time_seconds": asset["frame_time_seconds"],
                "viewpoint": asset["viewpoint"],
                "box_count": len(annotation["boxes"]),
                "verified_negative": asset["review_status"] == "negative",
            }
        )
        license_key = (
            str(asset["rights_statement"] or ""),
            str(asset["rights_url"] or ""),
            str(asset["creator_credit"] or ""),
        )
        entry = licenses.setdefault(
            license_key,
            {
                "rights_statement": license_key[0],
                "rights_url": license_key[1],
                "creator_credit": license_key[2],
                "asset_count": "0",
            },
        )
        entry["asset_count"] = str(int(entry["asset_count"]) + 1)

    with (root / "provenance.jsonl").open("w", encoding="utf-8") as fp:
        for row in manifest_rows:
            fp.write(json.dumps(row, sort_keys=True) + "\n")
    with (root / "licenses.csv").open("w", encoding="utf-8", newline="") as fp:
        fields = ["rights_statement", "rights_url", "creator_credit", "asset_count"]
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(licenses.values(), key=lambda value: tuple(value[field] for field in fields)))
    shutil.copy2(taxonomy.path, root / "taxonomy.yaml")
    (root / "taxonomy_version.txt").write_text(f"{taxonomy.name} {taxonomy.version}\n", encoding="utf-8")
    _write_data_yaml(root, taxonomy, root)
    rejection_counts = _rejection_counts(store)
    warn_below = int((settings.dataset_config.get("validation") or {}).get("warn_below_class_instances", 20))
    stats = {
        "assets": len(manifest_rows),
        "boxes": sum(class_boxes.values()),
        "images_by_split": dict(sorted(by_split.items())),
        "boxes_per_class": {name: class_boxes.get(name, 0) for name in taxonomy.names},
        "assets_by_source": dict(sorted(by_source.items())),
        "assets_by_viewpoint": dict(sorted(by_viewpoint.items())),
        "rejection_counts": rejection_counts,
        "underrepresented_warning_threshold": warn_below,
        "underrepresented_classes": [name for name in taxonomy.names if class_boxes.get(name, 0) < warn_below],
    }
    (root / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    build_config = {
        "dataset_name": settings.dataset_name,
        "version": version,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "taxonomy": {"name": taxonomy.name, "version": taxonomy.version},
        "split": {"method": "deterministic source-group assignment", "seed": seed, **proportions},
        "source_database": str(settings.database_path),
    }
    (root / "build_config.yaml").write_text(yaml.safe_dump(build_config, sort_keys=False), encoding="utf-8")
    (root / "DATASET_CARD.md").write_text(_dataset_card(build_config, stats), encoding="utf-8")
    return {"assets": len(manifest_rows), "boxes": sum(class_boxes.values()), "splits": dict(by_split), "version": version}


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _write_data_yaml(root: Path, taxonomy: Taxonomy, final_root: Path) -> None:
    payload = {
        "path": str(final_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(taxonomy.names)},
        "nc": len(taxonomy.names),
        "taxonomy_version": taxonomy.version,
    }
    (root / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _rejection_counts(store: MetadataStore) -> dict[str, int]:
    rows = store.list_assets()
    counts: Counter[str] = Counter()
    for row in rows:
        if row["download_status"] == "failed":
            counts["download_or_verification_failed"] += 1
        if row["permission_decision"] != "approved":
            counts[f"rights_{row['permission_decision']}"] += 1
        if row["review_status"] == "rejected":
            counts["human_rejected"] += 1
        if row["review_status"] == "ambiguous":
            counts["human_ambiguous"] += 1
        if row["review_status"] == "pending":
            counts["not_human_reviewed"] += 1
        if row["duplicate_of"]:
            counts["duplicate_excluded"] += 1
        if row["review_status"] == "accepted" and row["annotation_status"] != "validated":
            counts["accepted_but_unannotated"] += 1
    return dict(sorted(counts.items()))


def _dataset_card(build_config: dict[str, Any], stats: dict[str, Any]) -> str:
    return f"""# {build_config['dataset_name']} {build_config['version']}

This is a human-reviewed object-detection dataset assembled by GarudaEdge's Part A
pipeline. It contains only items whose recorded reuse decision is `approved`, whose
visual review is `accepted` or `negative`, and whose annotation import passed coordinate,
taxonomy, stable-identity, and dimension checks.

It does **not** establish platform identity, affiliation, intent, or operational status.
Search terms and captions are retained only as provenance. Rights remain governed by the
per-item statements in `provenance.jsonl` and the attribution summary in `licenses.csv`.

## Counts

- Images: {stats['assets']}
- Boxes: {stats['boxes']}
- Splits: {json.dumps(stats['images_by_split'], sort_keys=True)}
- Boxes per class: {json.dumps(stats['boxes_per_class'], sort_keys=True)}
- Sources: {json.dumps(stats['assets_by_source'], sort_keys=True)}
- Viewpoints: {json.dumps(stats['assets_by_viewpoint'], sort_keys=True)}

## Splitting and limitations

Splits are deterministic from seed {build_config['split']['seed']} and grouped by source
group, with near-duplicate keepers isolated. Video frames from one parent share a source
group. Small or visually ambiguous objects, geographic/source bias, license constraints,
and underrepresented classes require evaluation before any model use. This dataset is for
human-in-the-loop detection research; it is not an identity or intent dataset.
"""


def validate_dataset(
    dataset_root: str | Path,
    *,
    settings: DatasetSettings,
    taxonomy: Taxonomy,
    write_report: bool = True,
) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    manifest_path = root / "provenance.jsonl"
    data_yaml_path = root / "data.yaml"
    required_artifacts = (
        "licenses.csv", "DATASET_CARD.md", "taxonomy.yaml", "taxonomy_version.txt",
        "stats.json", "build_config.yaml",
    )
    for artifact in required_artifacts:
        if not (root / artifact).is_file():
            errors.append(f"missing {artifact}")
    if not data_yaml_path.exists():
        errors.append("missing data.yaml")
    else:
        try:
            data = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8")) or {}
            names = data.get("names") or {}
            actual_names = [str(names[key]) for key in sorted(names, key=lambda value: int(value))] if isinstance(names, dict) else list(names)
            if actual_names != taxonomy.names:
                errors.append("data.yaml class names do not match taxonomy")
            if int(data.get("nc", -1)) != len(taxonomy.names):
                errors.append("data.yaml nc does not match taxonomy")
            configured_root = Path(str(data.get("path") or "")).expanduser().resolve()
            if configured_root != root:
                errors.append(f"data.yaml path points to {configured_root}, expected {root}")
        except Exception as exc:
            errors.append(f"malformed data.yaml: {exc}")
    rows: list[dict[str, Any]] = []
    if not manifest_path.exists():
        errors.append("missing provenance.jsonl")
    else:
        for number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"provenance.jsonl line {number}: {exc}")
    sha_splits: dict[str, set[str]] = defaultdict(set)
    sha_assets: dict[str, list[str]] = defaultdict(list)
    group_splits: dict[str, set[str]] = defaultdict(set)
    cluster_splits: dict[str, set[str]] = defaultdict(set)
    class_counts: Counter[int] = Counter()
    split_counts: Counter[str] = Counter()
    for row in rows:
        asset_id = str(row.get("asset_id") or "<missing>")
        split = str(row.get("split") or "")
        if split not in SPLITS:
            errors.append(f"{asset_id}: invalid split {split!r}")
            continue
        split_counts[split] += 1
        image_path = root / str(row.get("file") or "")
        label_path = root / str(row.get("label_file") or "")
        if not image_path.is_file():
            errors.append(f"{asset_id}: missing image {image_path}")
        else:
            if sha256_file(image_path) != row.get("sha256"):
                errors.append(f"{asset_id}: SHA-256 mismatch")
            try:
                if _image_size(image_path) != (int(row.get("width")), int(row.get("height"))):
                    errors.append(f"{asset_id}: image dimensions differ from provenance")
            except Exception as exc:
                errors.append(f"{asset_id}: image decode/dimension check failed: {exc}")
        if not label_path.is_file():
            errors.append(f"{asset_id}: missing label {label_path}")
        else:
            label_boxes, label_errors = _validate_label(label_path, taxonomy.ids)
            errors.extend(f"{asset_id}: {message}" for message in label_errors)
            for class_id in label_boxes:
                class_counts[class_id] += 1
            if row.get("review_status") == "accepted" and not label_boxes:
                errors.append(f"{asset_id}: unreviewed/empty positive label")
            if row.get("verified_negative") and label_boxes:
                errors.append(f"{asset_id}: verified negative has boxes")
        if row.get("permission_decision") != "approved" or not row.get("rights_statement"):
            errors.append(f"{asset_id}: unresolved rights")
        if row.get("review_status") not in {"accepted", "negative"}:
            errors.append(f"{asset_id}: not human reviewed")
        if row.get("annotation_status") != "validated":
            errors.append(f"{asset_id}: annotation not validated")
        sha_splits[str(row.get("sha256"))].add(split)
        sha_assets[str(row.get("sha256"))].append(asset_id)
        group_splits[str(row.get("source_group_id"))].add(split)
        if row.get("near_duplicate_cluster"):
            cluster_splits[str(row["near_duplicate_cluster"])].add(split)
    for digest, splits in sha_splits.items():
        if len(splits) > 1:
            errors.append(f"duplicate content {digest} crosses splits {sorted(splits)}")
        if len(sha_assets[digest]) > 1:
            errors.append(f"duplicate content {digest} appears in assets {sorted(sha_assets[digest])}")
    for group, splits in group_splits.items():
        if len(splits) > 1:
            errors.append(f"source group {group!r} crosses splits {sorted(splits)}")
    for cluster, splits in cluster_splits.items():
        if len(splits) > 1:
            errors.append(f"near-duplicate cluster {cluster!r} crosses splits {sorted(splits)}")
    if len(group_splits) >= len(SPLITS):
        for split in SPLITS:
            if not split_counts[split]:
                errors.append(f"split {split} is empty despite at least three independent groups")
    minimum = int((settings.dataset_config.get("validation") or {}).get("minimum_class_instances", 1))
    warn_below = int((settings.dataset_config.get("validation") or {}).get("warn_below_class_instances", 20))
    for class_id, name in enumerate(taxonomy.names):
        count = class_counts[class_id]
        if count < minimum:
            errors.append(f"class {class_id} ({name}) has {count} instances; minimum is {minimum}")
        elif count < warn_below:
            warnings.append(f"class {class_id} ({name}) is under the warning threshold: {count} < {warn_below}")
    report = {
        "valid": not errors,
        "dataset_root": str(root),
        "assets": len(rows),
        "boxes": sum(class_counts.values()),
        "splits": dict(split_counts),
        "class_instances": {taxonomy.names[key]: class_counts[key] for key in range(len(taxonomy.names))},
        "errors": errors,
        "warnings": warnings,
        "checks": [
            "missing images and labels", "class IDs", "box bounds and area", "content hash leakage",
            "source-group leakage", "near-duplicate-cluster leakage", "rights", "human review",
            "annotation validation", "empty splits", "empty/underrepresented classes",
        ],
    }
    if write_report:
        root.mkdir(parents=True, exist_ok=True)
        (root / "validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _validate_label(path: Path, valid_ids: set[int]) -> tuple[list[int], list[str]]:
    classes: list[int] = []
    errors: list[str] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 5:
            errors.append(f"label line {number} has {len(fields)} fields")
            continue
        try:
            class_id = int(fields[0])
            x, y, width, height = (float(value) for value in fields[1:])
        except ValueError:
            errors.append(f"label line {number} contains non-numeric values")
            continue
        if class_id not in valid_ids:
            errors.append(f"label line {number} has invalid class ID {class_id}")
            continue
        values = (x, y, width, height)
        if not all(value == value and abs(value) != float("inf") for value in values):
            errors.append(f"label line {number} has non-finite coordinates")
        elif width <= 0 or height <= 0:
            errors.append(f"label line {number} has zero or negative area")
        elif not all(0 <= value <= 1 for value in values):
            errors.append(f"label line {number} is outside normalized bounds")
        elif x - width / 2 < -1e-8 or y - height / 2 < -1e-8 or x + width / 2 > 1 + 1e-8 or y + height / 2 > 1 + 1e-8:
            errors.append(f"label line {number} extends outside image bounds")
        else:
            classes.append(class_id)
    return classes, errors


def _image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Dataset validation requires Pillow") from exc
    with Image.open(path) as image:
        image.load()
        return image.size
