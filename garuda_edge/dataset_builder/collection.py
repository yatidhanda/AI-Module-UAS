from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import DatasetSettings
from .http import PoliteHTTPClient
from .providers import (
    BraveImageProvider,
    DVIDSProvider,
    LocalImportProvider,
    SerpAPIImageProvider,
    WikimediaCommonsProvider,
)
from .store import MetadataStore
from .taxonomy import Taxonomy


PROVIDERS = {
    "wikimedia": WikimediaCommonsProvider,
    "dvids": DVIDSProvider,
    "local": LocalImportProvider,
    "brave": BraveImageProvider,
    "serpapi": SerpAPIImageProvider,
}


def make_client(settings: DatasetSettings) -> PoliteHTTPClient:
    media = settings.dataset_config.get("media") or {}
    return PoliteHTTPClient(
        user_agent=(
            "GarudaEdgeDataset/0.1 "
            "(+https://github.com/yatidhanda/AI-Module-UAS; provenance-preserving research collector)"
        ),
        timeout_seconds=float(media.get("request_timeout_seconds", 30)),
        requests_per_second=float(media.get("requests_per_second", 1.0)),
        retries=3,
        backoff_seconds=1.0,
    )


def discover_assets(
    settings: DatasetSettings,
    taxonomy: Taxonomy,
    *,
    provider_name: str,
    queries: list[str],
    category: str | None,
    limit: int,
    media_types: set[str],
    viewpoint: str | None = None,
    dry_run: bool = False,
    provider_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if provider_name not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider_name!r}; choose from {', '.join(sorted(PROVIDERS))}")
    settings.ensure_layout()
    store = MetadataStore(settings.database_path)
    provider = PROVIDERS[provider_name](make_client(settings), **(provider_options or {}))
    discovered = created = existing = 0
    previews: list[dict[str, Any]] = []
    if not queries:
        raise ValueError("At least one search query is required")
    per_query = max(1, limit // len(queries))
    remainder = max(0, limit - per_query * len(queries))
    for index, query in enumerate(queries):
        query_limit = per_query + (1 if index < remainder else 0)
        candidate_category = category or taxonomy.candidate_for_term(query)
        for candidate in provider.discover(
            query=query,
            candidate_category=candidate_category,
            limit=query_limit,
            media_types=media_types,
            viewpoint=viewpoint,
        ):
            if discovered >= limit:
                break
            discovered += 1
            if len(previews) < 20:
                previews.append(candidate.to_dict())
            was_created = store.upsert_candidate(candidate, dry_run=dry_run)
            created += int(was_created)
            existing += int(not was_created and not dry_run)
        if discovered >= limit:
            break
    if not dry_run:
        write_workspace_manifests(settings, store)
    return {
        "provider": provider_name,
        "queries": queries,
        "discovered": discovered,
        "created": created,
        "already_known": existing,
        "dry_run": dry_run,
        "preview": previews,
    }


def profile_queries(taxonomy: Taxonomy, profile_name: str) -> tuple[list[str], str | None]:
    profile = taxonomy.search_profiles.get(profile_name)
    if not profile:
        available = ", ".join(sorted(taxonomy.search_profiles)) or "none"
        raise ValueError(f"Unknown search profile {profile_name!r}; available: {available}")
    viewpoints = profile.get("viewpoint") or []
    viewpoint = ",".join(str(item) for item in viewpoints) or None
    return [str(item) for item in (profile.get("queries") or [])], viewpoint


def write_workspace_manifests(settings: DatasetSettings, store: MetadataStore) -> tuple[Path, Path]:
    settings.reports_root.mkdir(parents=True, exist_ok=True)
    assets = store.list_assets()
    queries = store.discovery_queries()
    for asset in assets:
        asset["discovery_queries"] = queries.get(str(asset["asset_id"]), [])
    jsonl_path = settings.reports_root / "candidate_assets.jsonl"
    csv_path = settings.reports_root / "candidate_assets.csv"
    with jsonl_path.open("w", encoding="utf-8") as fp:
        for asset in assets:
            fp.write(json.dumps(asset, sort_keys=True) + "\n")
    fields = [
        "asset_id", "provider", "provider_id", "source_page_url", "media_url",
        "retrieval_timestamp", "creator_credit", "rights_statement", "rights_url",
        "permission_decision", "original_search_term", "candidate_category", "review_status",
        "download_status", "failure_reason", "sha256", "perceptual_hash", "mime_type",
        "width", "height", "duration_seconds", "fps", "byte_size", "local_path",
        "source_group_id", "parent_asset_id", "frame_time_seconds", "derived_frame_ids_json",
        "near_duplicate_cluster", "duplicate_of", "viewpoint", "annotation_status",
        "annotation_version", "reviewer",
        "discovery_queries",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for asset in assets:
            row = dict(asset)
            row["discovery_queries"] = json.dumps(row["discovery_queries"], sort_keys=True)
            writer.writerow(row)
    return jsonl_path, csv_path
