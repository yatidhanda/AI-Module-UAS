from __future__ import annotations

import html
import re
from collections.abc import Iterator
from typing import Any

from ..models import CandidateAsset, utc_now
from .base import DiscoveryProvider, stable_asset_id


API_URL = "https://commons.wikimedia.org/w/api.php"
_FREE_LICENSE_TOKENS = (
    "public domain",
    "cc0",
    "creativecommons.org/publicdomain/",
    "cc by 1.",
    "cc by 2.",
    "cc by 3.",
    "cc by 4.",
    "cc-by-",
    "cc by-sa",
    "cc-by-sa",
    "creative commons attribution",
    "gfdl",
)
_EXCLUDED_MIME = {"image/svg+xml", "application/pdf", "image/vnd.djvu"}


def _metadata_value(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if isinstance(value, dict):
        value = value.get("value")
    if value is None:
        return None
    clean = re.sub(r"<[^>]+>", " ", html.unescape(str(value)))
    clean = " ".join(clean.split())
    return clean or None


def permission_from_commons(license_name: str | None, license_url: str | None) -> str:
    text = f"{license_name or ''} {license_url or ''}".lower()
    if any(token in text for token in ("by-nc", "by-nd", "noncommercial", "no derivatives", "no-derivatives")):
        return "pending"
    if any(token in text for token in _FREE_LICENSE_TOKENS):
        return "approved"
    return "pending"


class WikimediaCommonsProvider(DiscoveryProvider):
    name = "wikimedia"

    def discover(
        self,
        *,
        query: str,
        candidate_category: str,
        limit: int,
        media_types: set[str],
        viewpoint: str | None = None,
    ) -> Iterator[CandidateAsset]:
        remaining = max(0, limit)
        continuation: dict[str, Any] = {}
        while remaining:
            batch = min(50, remaining)
            params: dict[str, Any] = {
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "generator": "search",
                "gsrnamespace": 6,
                "gsrsearch": query,
                "gsrlimit": batch,
                "prop": "imageinfo|info",
                "iiprop": "url|mime|size|extmetadata|mediatype",
                "inprop": "url",
                "maxlag": 5,
            }
            params.update(continuation)
            payload = self.client.get(API_URL, params=params).json()
            pages = (payload.get("query") or {}).get("pages") or []
            if not pages:
                return
            yielded = 0
            for page in pages:
                info = (page.get("imageinfo") or [{}])[0]
                mime = str(info.get("mime") or "").lower()
                media_type = "video" if mime.startswith("video/") else "image"
                if mime in _EXCLUDED_MIME or media_type not in media_types:
                    continue
                metadata = info.get("extmetadata") or {}
                license_name = _metadata_value(metadata, "LicenseShortName") or _metadata_value(metadata, "UsageTerms")
                license_url = _metadata_value(metadata, "LicenseUrl")
                page_id = str(page.get("pageid") or page.get("title"))
                source_url = str(page.get("canonicalurl") or page.get("fullurl") or "")
                media_url = str(info.get("url") or "")
                if not source_url or not media_url:
                    continue
                yielded += 1
                yield CandidateAsset(
                    asset_id=stable_asset_id(self.name, page_id),
                    provider=self.name,
                    provider_id=page_id,
                    source_page_url=source_url,
                    media_url=media_url,
                    media_type=media_type,
                    retrieval_timestamp=utc_now(),
                    creator_credit=_metadata_value(metadata, "Artist") or _metadata_value(metadata, "Credit"),
                    rights_statement=license_name,
                    rights_url=license_url,
                    permission_decision=permission_from_commons(license_name, license_url),
                    original_search_term=query,
                    candidate_category=candidate_category,
                    source_group_id=f"wikimedia:{page_id}",
                    mime_type=mime or None,
                    width=int(info["width"]) if info.get("width") else None,
                    height=int(info["height"]) if info.get("height") else None,
                    viewpoint=viewpoint,
                    metadata={
                        "title": page.get("title"),
                        "description": _metadata_value(metadata, "ImageDescription"),
                        "attribution_required": _metadata_value(metadata, "AttributionRequired"),
                        "copyrighted": _metadata_value(metadata, "Copyrighted"),
                        "api": API_URL,
                    },
                )
            remaining -= yielded
            continuation = payload.get("continue") or {}
            if not continuation or yielded == 0:
                return
