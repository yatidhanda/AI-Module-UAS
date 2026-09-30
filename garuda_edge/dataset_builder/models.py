from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


PERMISSION_DECISIONS = {"pending", "approved", "rejected"}
REVIEW_STATUSES = {"pending", "accepted", "rejected", "ambiguous", "negative"}
DOWNLOAD_STATUSES = {"discovered", "downloaded", "failed", "quarantined", "local"}
ANNOTATION_STATUSES = {"unannotated", "validated", "rejected"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class CandidateAsset:
    asset_id: str
    provider: str
    provider_id: str
    source_page_url: str
    media_url: str
    media_type: str
    retrieval_timestamp: str
    creator_credit: str | None
    rights_statement: str | None
    rights_url: str | None
    permission_decision: str
    original_search_term: str
    candidate_category: str
    source_group_id: str
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    fps: float | None = None
    viewpoint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BoundingBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_yolo_line(self) -> str:
        return f"{self.class_id} {self.x_center:.8f} {self.y_center:.8f} {self.width:.8f} {self.height:.8f}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
