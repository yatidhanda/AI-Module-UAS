from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from .models import CandidateAsset, utc_now


SCHEMA_VERSION = 1


class MetadataStore:
    """Durable local metadata store.

    SQLite is the authority for workflow state; JSONL/CSV files written by report and
    build commands are portable snapshots. Originals are never deleted by this class.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_info (
                    version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS assets (
                    asset_id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    provider_id TEXT NOT NULL,
                    source_page_url TEXT NOT NULL,
                    media_url TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    retrieval_timestamp TEXT NOT NULL,
                    creator_credit TEXT,
                    rights_statement TEXT,
                    rights_url TEXT,
                    permission_decision TEXT NOT NULL DEFAULT 'pending',
                    original_search_term TEXT NOT NULL,
                    candidate_category TEXT NOT NULL,
                    review_status TEXT NOT NULL DEFAULT 'pending',
                    download_status TEXT NOT NULL DEFAULT 'discovered',
                    failure_reason TEXT,
                    sha256 TEXT,
                    perceptual_hash TEXT,
                    mime_type TEXT,
                    width INTEGER,
                    height INTEGER,
                    duration_seconds REAL,
                    fps REAL,
                    byte_size INTEGER,
                    local_path TEXT,
                    source_group_id TEXT NOT NULL,
                    parent_asset_id TEXT REFERENCES assets(asset_id),
                    frame_time_seconds REAL,
                    derived_frame_ids_json TEXT NOT NULL DEFAULT '[]',
                    near_duplicate_cluster TEXT,
                    duplicate_of TEXT REFERENCES assets(asset_id),
                    viewpoint TEXT,
                    annotation_status TEXT NOT NULL DEFAULT 'unannotated',
                    annotation_version TEXT,
                    reviewer TEXT,
                    annotation_imported_at TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(provider, provider_id)
                );
                CREATE INDEX IF NOT EXISTS idx_assets_workflow
                    ON assets(download_status, permission_decision, review_status, annotation_status);
                CREATE INDEX IF NOT EXISTS idx_assets_hash ON assets(sha256);
                CREATE INDEX IF NOT EXISTS idx_assets_group ON assets(source_group_id);
                CREATE TABLE IF NOT EXISTS asset_queries (
                    asset_id TEXT NOT NULL REFERENCES assets(asset_id),
                    provider TEXT NOT NULL,
                    search_term TEXT NOT NULL,
                    candidate_category TEXT NOT NULL,
                    discovered_at TEXT NOT NULL,
                    PRIMARY KEY(asset_id, provider, search_term)
                );
                CREATE TABLE IF NOT EXISTS annotations (
                    asset_id TEXT PRIMARY KEY REFERENCES assets(asset_id),
                    taxonomy_name TEXT NOT NULL,
                    taxonomy_version TEXT NOT NULL,
                    boxes_json TEXT NOT NULL,
                    annotation_version TEXT NOT NULL,
                    reviewer TEXT,
                    source_export TEXT NOT NULL,
                    imported_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS workflow_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset_id TEXT REFERENCES assets(asset_id),
                    event_type TEXT NOT NULL,
                    reason TEXT,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                """
            )
            row = db.execute("SELECT version FROM schema_info LIMIT 1").fetchone()
            if row is None:
                db.execute("INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,))
            elif int(row["version"]) != SCHEMA_VERSION:
                raise RuntimeError(
                    f"Unsupported metadata schema {row['version']}; expected {SCHEMA_VERSION}"
                )

    def upsert_candidate(self, candidate: CandidateAsset, *, dry_run: bool = False) -> bool:
        if dry_run:
            return False
        now = utc_now()
        values = candidate.to_dict()
        metadata_json = json.dumps(values.pop("metadata"), sort_keys=True)
        with self.connect() as db:
            existing = db.execute(
                "SELECT asset_id FROM assets WHERE provider=? AND provider_id=?",
                (candidate.provider, candidate.provider_id),
            ).fetchone()
            if existing:
                asset_id = str(existing["asset_id"])
                db.execute(
                    """
                    UPDATE assets SET source_page_url=?, media_url=?, retrieval_timestamp=?,
                        creator_credit=COALESCE(?, creator_credit),
                        rights_statement=COALESCE(?, rights_statement),
                        rights_url=COALESCE(?, rights_url),
                        permission_decision=CASE WHEN permission_decision='pending' THEN ? ELSE permission_decision END,
                        candidate_category=CASE WHEN candidate_category IN ('unknown','needs_review') THEN ? ELSE candidate_category END,
                        mime_type=COALESCE(?, mime_type), width=COALESCE(?, width),
                        height=COALESCE(?, height), duration_seconds=COALESCE(?, duration_seconds),
                        fps=COALESCE(?, fps), viewpoint=COALESCE(?, viewpoint),
                        metadata_json=?, updated_at=? WHERE asset_id=?
                    """,
                    (
                        candidate.source_page_url,
                        candidate.media_url,
                        candidate.retrieval_timestamp,
                        candidate.creator_credit,
                        candidate.rights_statement,
                        candidate.rights_url,
                        candidate.permission_decision,
                        candidate.candidate_category,
                        candidate.mime_type,
                        candidate.width,
                        candidate.height,
                        candidate.duration_seconds,
                        candidate.fps,
                        candidate.viewpoint,
                        metadata_json,
                        now,
                        asset_id,
                    ),
                )
                created = False
            else:
                asset_id = candidate.asset_id
                db.execute(
                    """
                    INSERT INTO assets (
                        asset_id, provider, provider_id, source_page_url, media_url, media_type,
                        retrieval_timestamp, creator_credit, rights_statement, rights_url,
                        permission_decision, original_search_term, candidate_category,
                        mime_type, width, height, duration_seconds, fps, source_group_id,
                        viewpoint, metadata_json, created_at, updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        asset_id,
                        candidate.provider,
                        candidate.provider_id,
                        candidate.source_page_url,
                        candidate.media_url,
                        candidate.media_type,
                        candidate.retrieval_timestamp,
                        candidate.creator_credit,
                        candidate.rights_statement,
                        candidate.rights_url,
                        candidate.permission_decision,
                        candidate.original_search_term,
                        candidate.candidate_category,
                        candidate.mime_type,
                        candidate.width,
                        candidate.height,
                        candidate.duration_seconds,
                        candidate.fps,
                        candidate.source_group_id,
                        candidate.viewpoint,
                        metadata_json,
                        now,
                        now,
                    ),
                )
                created = True
            db.execute(
                """
                INSERT OR IGNORE INTO asset_queries
                    (asset_id, provider, search_term, candidate_category, discovered_at)
                VALUES (?,?,?,?,?)
                """,
                (asset_id, candidate.provider, candidate.original_search_term, candidate.candidate_category, now),
            )
        return created

    def add_derived_asset(self, values: dict[str, Any]) -> bool:
        now = utc_now()
        columns = [
            "asset_id", "provider", "provider_id", "source_page_url", "media_url", "media_type",
            "retrieval_timestamp", "creator_credit", "rights_statement", "rights_url",
            "permission_decision", "original_search_term", "candidate_category", "review_status",
            "download_status", "sha256", "perceptual_hash", "mime_type", "width", "height",
            "byte_size", "local_path", "source_group_id", "parent_asset_id", "frame_time_seconds",
            "viewpoint", "metadata_json", "created_at", "updated_at",
        ]
        payload = dict(values)
        payload.setdefault("review_status", "pending")
        payload.setdefault("download_status", "downloaded")
        payload.setdefault("metadata_json", "{}")
        if isinstance(payload["metadata_json"], dict):
            payload["metadata_json"] = json.dumps(payload["metadata_json"], sort_keys=True)
        payload["created_at"] = now
        payload["updated_at"] = now
        with self.connect() as db:
            cursor = db.execute(
                f"INSERT OR IGNORE INTO assets ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                tuple(payload.get(column) for column in columns),
            )
            if cursor.rowcount:
                parent_id = str(payload["parent_asset_id"])
                child_ids = [
                    row["asset_id"]
                    for row in db.execute("SELECT asset_id FROM assets WHERE parent_asset_id=? ORDER BY asset_id", (parent_id,))
                ]
                db.execute(
                    "UPDATE assets SET derived_frame_ids_json=?, updated_at=? WHERE asset_id=?",
                    (json.dumps(child_ids), now, parent_id),
                )
                return True
            return False

    def get_asset(self, asset_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM assets WHERE asset_id=?", (asset_id,)).fetchone()
        return dict(row) if row else None

    def find_asset(self, provider: str, provider_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM assets WHERE provider=? AND provider_id=?", (provider, provider_id)
            ).fetchone()
        return dict(row) if row else None

    def list_assets(
        self,
        *,
        where: str = "1=1",
        params: Sequence[Any] = (),
        limit: int | None = None,
        order_by: str = "created_at, asset_id",
    ) -> list[dict[str, Any]]:
        sql = f"SELECT * FROM assets WHERE {where} ORDER BY {order_by}"
        query_params: list[Any] = list(params)
        if limit is not None:
            sql += " LIMIT ?"
            query_params.append(limit)
        with self.connect() as db:
            rows = db.execute(sql, query_params).fetchall()
        return [dict(row) for row in rows]

    def update_asset(self, asset_id: str, **fields: Any) -> None:
        if not fields:
            return
        allowed = {
            "permission_decision", "review_status", "download_status", "failure_reason", "sha256",
            "perceptual_hash", "mime_type", "width", "height", "duration_seconds", "fps",
            "byte_size", "local_path", "near_duplicate_cluster", "duplicate_of", "viewpoint",
            "annotation_status", "annotation_version", "reviewer", "annotation_imported_at",
            "metadata_json", "rights_statement", "rights_url", "creator_credit",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Unsupported asset fields: {sorted(unknown)}")
        if "metadata_json" in fields and isinstance(fields["metadata_json"], dict):
            fields["metadata_json"] = json.dumps(fields["metadata_json"], sort_keys=True)
        fields["updated_at"] = utc_now()
        assignments = ", ".join(f"{key}=?" for key in fields)
        with self.connect() as db:
            cursor = db.execute(
                f"UPDATE assets SET {assignments} WHERE asset_id=?",
                (*fields.values(), asset_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown asset ID: {asset_id}")

    def record_event(
        self,
        event_type: str,
        *,
        asset_id: str | None = None,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO workflow_events(asset_id,event_type,reason,details_json,created_at) VALUES(?,?,?,?,?)",
                (asset_id, event_type, reason, json.dumps(details or {}, sort_keys=True), utc_now()),
            )

    def has_event(self, asset_id: str, event_type: str) -> bool:
        with self.connect() as db:
            row = db.execute(
                "SELECT 1 FROM workflow_events WHERE asset_id=? AND event_type=? LIMIT 1",
                (asset_id, event_type),
            ).fetchone()
        return row is not None

    def set_annotation(
        self,
        *,
        asset_id: str,
        taxonomy_name: str,
        taxonomy_version: str,
        boxes: list[dict[str, Any]],
        annotation_version: str,
        reviewer: str | None,
        source_export: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        imported_at = utc_now()
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO annotations(asset_id,taxonomy_name,taxonomy_version,boxes_json,
                    annotation_version,reviewer,source_export,imported_at,metadata_json)
                VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(asset_id) DO UPDATE SET
                    taxonomy_name=excluded.taxonomy_name,
                    taxonomy_version=excluded.taxonomy_version,
                    boxes_json=excluded.boxes_json,
                    annotation_version=excluded.annotation_version,
                    reviewer=excluded.reviewer,
                    source_export=excluded.source_export,
                    imported_at=excluded.imported_at,
                    metadata_json=excluded.metadata_json
                """,
                (
                    asset_id,
                    taxonomy_name,
                    taxonomy_version,
                    json.dumps(boxes, sort_keys=True),
                    annotation_version,
                    reviewer,
                    source_export,
                    imported_at,
                    json.dumps(metadata or {}, sort_keys=True),
                ),
            )
            db.execute(
                """
                UPDATE assets SET annotation_status='validated', annotation_version=?, reviewer=?,
                    annotation_imported_at=?, updated_at=? WHERE asset_id=?
                """,
                (annotation_version, reviewer, imported_at, imported_at, asset_id),
            )

    def get_annotation(self, asset_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM annotations WHERE asset_id=?", (asset_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["boxes"] = json.loads(result.pop("boxes_json"))
        result["metadata"] = json.loads(result.pop("metadata_json"))
        return result

    def counts(self) -> dict[str, Any]:
        with self.connect() as db:
            total = int(db.execute("SELECT COUNT(*) FROM assets").fetchone()[0])
            result: dict[str, Any] = {"discovered": total}
            for field in (
                "provider", "media_type", "download_status", "permission_decision",
                "review_status", "annotation_status", "candidate_category",
            ):
                result[f"by_{field}"] = {
                    str(row[0]): int(row[1])
                    for row in db.execute(f"SELECT {field}, COUNT(*) FROM assets GROUP BY {field}")
                }
            result["downloaded"] = int(
                db.execute("SELECT COUNT(*) FROM assets WHERE download_status IN ('downloaded','local','quarantined')").fetchone()[0]
            )
            result["rights_approved"] = int(
                db.execute("SELECT COUNT(*) FROM assets WHERE permission_decision='approved'").fetchone()[0]
            )
            result["deduplicated_unique"] = int(
                db.execute(
                    "SELECT COUNT(*) FROM assets WHERE download_status IN ('downloaded','local','quarantined') AND duplicate_of IS NULL"
                ).fetchone()[0]
            )
            result["human_reviewed"] = int(
                db.execute("SELECT COUNT(*) FROM assets WHERE review_status!='pending'").fetchone()[0]
            )
            result["annotated"] = int(
                db.execute("SELECT COUNT(*) FROM assets WHERE annotation_status='validated'").fetchone()[0]
            )
            result["training_ready"] = int(
                db.execute(
                    """
                    SELECT COUNT(*) FROM assets WHERE download_status IN ('downloaded','local')
                      AND permission_decision='approved' AND review_status IN ('accepted','negative')
                      AND annotation_status='validated' AND duplicate_of IS NULL
                    """
                ).fetchone()[0]
            )
        return result

    def discovery_queries(self) -> dict[str, list[dict[str, Any]]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT asset_id,provider,search_term,candidate_category,discovered_at "
                "FROM asset_queries ORDER BY asset_id,discovered_at,search_term"
            ).fetchall()
        result: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            payload = dict(row)
            result.setdefault(str(payload.pop("asset_id")), []).append(payload)
        return result
