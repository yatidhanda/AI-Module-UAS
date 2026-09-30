from __future__ import annotations

import mimetypes
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..models import CandidateAsset, utc_now
from .base import DiscoveryProvider, ProviderUnavailable, stable_asset_id


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".webm"}


class LocalImportProvider(DiscoveryProvider):
    name = "local"

    @property
    def available(self) -> bool:
        return bool(self.options.get("path"))

    def discover(
        self,
        *,
        query: str,
        candidate_category: str,
        limit: int,
        media_types: set[str],
        viewpoint: str | None = None,
    ) -> Iterator[CandidateAsset]:
        if not self.available:
            raise ProviderUnavailable("Local discovery requires --local-path")
        root = Path(str(self.options["path"])).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Local import path does not exist: {root}")
        rights = str(self.options.get("rights_statement") or "").strip()
        rights_url = str(self.options.get("rights_url") or "").strip() or None
        permission = str(self.options.get("permission_decision") or "pending")
        if not rights:
            raise ValueError("Local imports require an exact --rights-statement")
        paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        source_name = str(self.options.get("source_name") or root.name)
        count = 0
        for path in paths:
            suffix = path.suffix.lower()
            media_type = "image" if suffix in IMAGE_SUFFIXES else "video" if suffix in VIDEO_SUFFIXES else None
            if not media_type or media_type not in media_types:
                continue
            relative = path.name if root.is_file() else str(path.relative_to(root))
            provider_id = f"{source_name}:{relative}"
            yield CandidateAsset(
                asset_id=stable_asset_id(self.name, provider_id),
                provider=self.name,
                provider_id=provider_id,
                source_page_url=path.as_uri(),
                media_url=path.as_uri(),
                media_type=media_type,
                retrieval_timestamp=utc_now(),
                creator_credit=str(self.options.get("creator_credit") or "") or None,
                rights_statement=rights,
                rights_url=rights_url,
                permission_decision=permission,
                original_search_term=query or str(self.options.get("source_name") or "local import"),
                candidate_category=candidate_category,
                source_group_id=str(self.options.get("source_group_id") or f"local:{source_name}:{path.stem}"),
                mime_type=mimetypes.guess_type(path.name)[0],
                viewpoint=viewpoint,
                metadata={"original_path": str(path), "source_name": source_name},
            )
            count += 1
            if count >= limit:
                return
