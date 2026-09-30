from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

from ..models import CandidateAsset, utc_now
from .base import DiscoveryProvider, ProviderUnavailable, stable_asset_id


API_ROOT = "https://api.dvidshub.net"
COPYRIGHT_URL = "https://www.dvidshub.net/about/copyright"
DVIDS_RIGHTS_NOTICE = (
    "DVIDS states that many official-duty U.S. Government works are not eligible for "
    "U.S. copyright, but some hosted items have third-party rights. Item-level human "
    "rights review is required before dataset export."
)


class DVIDSProvider(DiscoveryProvider):
    name = "dvids"
    requires_credentials = True

    def __init__(self, client, **options: Any):
        super().__init__(client, **options)
        self.api_key = str(options.get("api_key") or os.environ.get("DVIDS_API_KEY") or "")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

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
            raise ProviderUnavailable(
                "DVIDS discovery requires DVIDS_API_KEY. Register under the DVIDS API terms; "
                "do not put the key in tracked configuration."
            )
        remaining = max(0, limit)
        requested_types = sorted(media_types & {"image", "video"})
        for media_type in requested_types:
            page = 1
            while remaining:
                batch = min(50, remaining)
                payload = self.client.get(
                    f"{API_ROOT}/search",
                    params={
                        "q": query,
                        "type": media_type,
                        "max_results": batch,
                        "page": page,
                        "api_key": self.api_key,
                    },
                ).json()
                results = payload.get("results") or []
                if not results:
                    break
                yielded = 0
                for result in results:
                    if str(result.get("type") or "") != media_type:
                        continue
                    candidate = self._candidate(
                        result,
                        media_type=media_type,
                        query=query,
                        candidate_category=candidate_category,
                        viewpoint=viewpoint,
                    )
                    if candidate is not None:
                        yielded += 1
                        yield candidate
                remaining -= yielded
                page += 1
                if len(results) < batch or yielded == 0:
                    break

    def _candidate(
        self,
        result: dict[str, Any],
        *,
        media_type: str,
        query: str,
        candidate_category: str,
        viewpoint: str | None,
    ) -> CandidateAsset | None:
        provider_id = str(result.get("id") or "")
        if not provider_id:
            return None
        details = self.client.get(
            f"{API_ROOT}/asset",
            params={"id": provider_id, "api_key": self.api_key, "include_related_media": 1},
        ).json().get("results") or {}
        media_url = self._media_url(details, media_type)
        if not media_url:
            return None
        credit = details.get("credit") or result.get("credit")
        if isinstance(credit, list):
            credit = "; ".join(
                " ".join(filter(None, [str(item.get("rank") or ""), str(item.get("name") or "")])).strip()
                for item in credit
            )
        dimensions = details.get("dimensions") or {}
        related_ids = [
            str(item.get("id"))
            for item in (details.get("related_media") or [])
            if isinstance(item, dict) and item.get("id")
        ]
        group_members = sorted({provider_id, *related_ids})
        source_group_id = (
            f"dvids:related:{group_members[0]}" if len(group_members) > 1 else f"dvids:{provider_id}"
        )
        return CandidateAsset(
            asset_id=stable_asset_id(self.name, provider_id),
            provider=self.name,
            provider_id=provider_id,
            source_page_url=str(details.get("url") or result.get("url") or ""),
            media_url=media_url,
            media_type=media_type,
            retrieval_timestamp=utc_now(),
            creator_credit=str(credit) if credit else None,
            rights_statement=DVIDS_RIGHTS_NOTICE,
            rights_url=COPYRIGHT_URL,
            permission_decision="pending",
            original_search_term=query,
            candidate_category=candidate_category,
            source_group_id=source_group_id,
            width=int(dimensions["width"]) if dimensions.get("width") else None,
            height=int(dimensions["height"]) if dimensions.get("height") else None,
            duration_seconds=float(details["duration"]) if details.get("duration") else None,
            viewpoint=viewpoint,
            metadata={
                "title": details.get("title") or result.get("title"),
                "description": details.get("description") or result.get("short_description"),
                "virin": details.get("virin"),
                "branch": details.get("branch"),
                "date": details.get("date"),
                "related_media_ids": related_ids,
                "rights_review_note": "DVIDS domain is not proof of public-domain status.",
                "api_asset_last_checked": utc_now(),
            },
        )

    @staticmethod
    def _media_url(details: dict[str, Any], media_type: str) -> str | None:
        if media_type == "image":
            return details.get("image")
        files = details.get("files") or []
        if isinstance(files, dict):
            files = list(files.values())
        choices: list[tuple[int, str]] = []
        for item in files:
            if not isinstance(item, dict):
                continue
            url = item.get("url") or item.get("file")
            if url:
                choices.append((int(item.get("height") or item.get("bitrate") or 0), str(url)))
        return max(choices, default=(0, ""))[1] or None
