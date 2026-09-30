from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .collection import write_workspace_manifests
from .config import DatasetSettings
from .media import hash_distance
from .models import PERMISSION_DECISIONS, REVIEW_STATUSES
from .store import MetadataStore


class _DisjointSet:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def deduplicate_assets(
    settings: DatasetSettings,
    *,
    threshold: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    store = MetadataStore(settings.database_path)
    rows = store.list_assets(
        where="media_type='image' AND sha256 IS NOT NULL AND perceptual_hash IS NOT NULL "
        "AND download_status IN ('downloaded','quarantined','local')"
    )
    configured = (settings.dataset_config.get("deduplication") or {}).get("perceptual_hash_distance", 6)
    distance = int(configured if threshold is None else threshold)
    dsu = _DisjointSet([str(row["asset_id"]) for row in rows])
    exact_pairs = near_pairs = 0
    by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sha[str(row["sha256"])].append(row)
    for items in by_sha.values():
        for other in items[1:]:
            dsu.union(str(items[0]["asset_id"]), str(other["asset_id"]))
            exact_pairs += 1

    # Deliberately simple for a local pilot. Exact hashes are collapsed first; pHash
    # comparisons only run across different content hashes.
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            if left["sha256"] == right["sha256"]:
                continue
            if hash_distance(str(left["perceptual_hash"]), str(right["perceptual_hash"])) <= distance:
                dsu.union(str(left["asset_id"]), str(right["asset_id"]))
                near_pairs += 1

    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[dsu.find(str(row["asset_id"]))].append(row)
    duplicate_clusters = [items for items in clusters.values() if len(items) > 1]
    report_clusters: list[dict[str, Any]] = []
    duplicate_count = 0
    if not dry_run:
        for row in rows:
            store.update_asset(str(row["asset_id"]), near_duplicate_cluster=None, duplicate_of=None)
    for index, items in enumerate(sorted(duplicate_clusters, key=lambda x: min(str(i["asset_id"]) for i in x)), 1):
        cluster_id = f"cluster-{index:06d}"
        ranked = sorted(items, key=_quality_key, reverse=True)
        keeper = ranked[0]
        members = []
        for item in ranked:
            duplicate_of = None if item is keeper else str(keeper["asset_id"])
            duplicate_count += int(duplicate_of is not None)
            members.append({"asset_id": item["asset_id"], "duplicate_of": duplicate_of, "path": item["local_path"]})
            if not dry_run:
                store.update_asset(
                    str(item["asset_id"]),
                    near_duplicate_cluster=cluster_id,
                    duplicate_of=duplicate_of,
                )
                if duplicate_of:
                    store.record_event("marked_duplicate", asset_id=str(item["asset_id"]), details={"cluster": cluster_id, "keeper": duplicate_of})
        report_clusters.append({"cluster_id": cluster_id, "keeper": keeper["asset_id"], "members": members})
    if not dry_run:
        for row in rows:
            store.record_event(
                "dedup_examined",
                asset_id=str(row["asset_id"]),
                details={"hash_type": "dHash-64", "threshold": distance},
            )
    report = {
        "examined": len(rows),
        "threshold": distance,
        "hash_type": "dHash-64",
        "exact_pairs": exact_pairs,
        "near_pairs": near_pairs,
        "clusters": len(duplicate_clusters),
        "duplicates_marked": duplicate_count,
        "dry_run": dry_run,
        "cluster_details": report_clusters,
        "note": "No original files were deleted.",
    }
    settings.reports_root.mkdir(parents=True, exist_ok=True)
    path = settings.reports_root / "deduplication_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(path)
    if not dry_run:
        write_workspace_manifests(settings, store)
    return report


def _quality_key(item: dict[str, Any]) -> tuple[int, int, int, str]:
    return (
        int(item.get("permission_decision") == "approved"),
        int(item.get("width") or 0) * int(item.get("height") or 0),
        int(item.get("byte_size") or 0),
        str(item.get("asset_id") or ""),
    )


def set_review_status(
    settings: DatasetSettings,
    asset_ids: list[str],
    *,
    status: str,
    reason: str | None = None,
    permission_decision: str | None = None,
    rights_statement: str | None = None,
    rights_url: str | None = None,
    reviewer: str | None = None,
) -> dict[str, Any]:
    if status not in REVIEW_STATUSES:
        raise ValueError(f"Invalid review status {status!r}; choose from {sorted(REVIEW_STATUSES)}")
    if permission_decision and permission_decision not in PERMISSION_DECISIONS:
        raise ValueError(f"Invalid permission decision {permission_decision!r}")
    if permission_decision == "approved" and not rights_statement:
        raise ValueError("Approving rights requires an exact --rights-statement")
    store = MetadataStore(settings.database_path)
    updated = 0
    for asset_id in asset_ids:
        asset = store.get_asset(asset_id)
        if not asset:
            raise KeyError(f"Unknown asset ID: {asset_id}")
        fields: dict[str, Any] = {"review_status": status}
        if reviewer:
            fields["reviewer"] = reviewer
        if permission_decision:
            fields["permission_decision"] = permission_decision
            fields["rights_statement"] = rights_statement
            fields["rights_url"] = rights_url
            if asset.get("local_path"):
                fields["download_status"] = "downloaded" if permission_decision == "approved" else "quarantined"
        store.update_asset(asset_id, **fields)
        store.record_event(
            "human_review",
            asset_id=asset_id,
            reason=reason,
            details={"status": status, "reviewer": reviewer, "permission_decision": permission_decision},
        )
        if permission_decision:
            for child in store.list_assets(where="parent_asset_id=?", params=(asset_id,)):
                child_fields: dict[str, Any] = {
                    "permission_decision": permission_decision,
                    "rights_statement": rights_statement,
                    "rights_url": rights_url,
                }
                if child.get("local_path"):
                    child_fields["download_status"] = "downloaded" if permission_decision == "approved" else "quarantined"
                store.update_asset(str(child["asset_id"]), **child_fields)
                store.record_event(
                    "derived_rights_inherited",
                    asset_id=str(child["asset_id"]),
                    details={"parent_asset_id": asset_id, "permission_decision": permission_decision},
                )
        updated += 1
    write_workspace_manifests(settings, store)
    return {"updated": updated, "status": status, "permission_decision": permission_decision}


def generate_review_report(settings: DatasetSettings, *, limit: int | None = None) -> dict[str, Any]:
    store = MetadataStore(settings.database_path)
    rows = store.list_assets(
        where="local_path IS NOT NULL AND media_type='image'",
        limit=limit,
        order_by="review_status, permission_decision, candidate_category, asset_id",
    )
    settings.reports_root.mkdir(parents=True, exist_ok=True)
    thumbnails = settings.reports_root / "review_thumbnails"
    thumbnails.mkdir(parents=True, exist_ok=True)
    cards = []
    for row in rows:
        thumb_path = thumbnails / f"{row['asset_id']}.jpg"
        try:
            _make_thumbnail(Path(str(row["local_path"])), thumb_path)
            image_src = f"review_thumbnails/{thumb_path.name}"
        except Exception:
            image_src = ""
        cards.append(
            "<article>"
            + (f'<img src="{html.escape(image_src)}" alt="{html.escape(str(row["asset_id"]))}">' if image_src else "<div>preview unavailable</div>")
            + f"<h2>{html.escape(str(row['asset_id']))}</h2>"
            + f"<p><b>Candidate:</b> {html.escape(str(row['candidate_category']))}</p>"
            + f"<p><b>Rights:</b> {html.escape(str(row['permission_decision']))} — {html.escape(str(row['rights_statement'] or 'missing'))}</p>"
            + f"<p><b>Review:</b> {html.escape(str(row['review_status']))}; <b>duplicate:</b> {html.escape(str(row['duplicate_of'] or 'no'))}</p>"
            + f'<p><a href="{html.escape(str(row["source_page_url"]))}">source page</a></p>'
            + f"<code>garuda-dataset review --asset-id {html.escape(str(row['asset_id']))} --status accepted --reviewer NAME</code>"
            + "</article>"
        )
    document = """<!doctype html><html><head><meta charset="utf-8"><title>GarudaEdge review</title>
<style>body{font-family:system-ui;margin:2rem}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:1rem}article{border:1px solid #bbb;padding:1rem}img{max-width:100%;height:220px;object-fit:contain;background:#eee}code{white-space:normal}</style>
</head><body><h1>GarudaEdge human review</h1><p>Automated candidate categories are not ground truth. Review visual content and rights separately. Originals have not been deleted.</p><main>""" + "\n".join(cards) + "</main></body></html>"
    html_path = settings.reports_root / "review.html"
    html_path.write_text(document, encoding="utf-8")
    csv_path = settings.reports_root / "review_queue.csv"
    fields = ["asset_id", "candidate_category", "permission_decision", "rights_statement", "review_status", "duplicate_of", "source_page_url", "local_path"]
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return {"assets": len(rows), "html": str(html_path), "csv": str(csv_path)}


def _make_thumbnail(source: Path, target: Path) -> None:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Review sheets require Pillow") from exc
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((640, 440))
        image.save(target, "JPEG", quality=85)


def dataset_stats(settings: DatasetSettings) -> dict[str, Any]:
    store = MetadataStore(settings.database_path)
    counts = store.counts()
    settings.reports_root.mkdir(parents=True, exist_ok=True)
    path = settings.reports_root / "stats.json"
    path.write_text(json.dumps(counts, indent=2, sort_keys=True), encoding="utf-8")
    counts["report_path"] = str(path)
    return counts
