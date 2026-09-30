from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

from ..models import CandidateAsset, utc_now
from .base import DiscoveryProvider, ProviderUnavailable, stable_asset_id


LEAD_RIGHTS = "Search result only; no reuse rights established. Human source-page rights review required."


class BraveImageProvider(DiscoveryProvider):
    name = "brave"
    requires_credentials = True

    def __init__(self, client, **options: Any):
        super().__init__(client, **options)
        self.api_key = str(options.get("api_key") or os.environ.get("BRAVE_SEARCH_API_KEY") or "")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def discover(self, *, query: str, candidate_category: str, limit: int, media_types: set[str], viewpoint: str | None = None) -> Iterator[CandidateAsset]:
        if not self.available:
            raise ProviderUnavailable("Brave discovery requires BRAVE_SEARCH_API_KEY")
        if "image" not in media_types:
            return
        payload = self.client.get(
            "https://api.search.brave.com/res/v1/images/search",
            params={"q": query, "count": min(100, limit), "safesearch": "strict"},
            headers={"X-Subscription-Token": self.api_key, "Accept": "application/json"},
        ).json()
        for item in (payload.get("results") or [])[:limit]:
            props = item.get("properties") or {}
            media_url = str(props.get("url") or item.get("thumbnail", {}).get("src") or "")
            source_url = str(item.get("url") or "")
            if not media_url or not source_url:
                continue
            provider_id = source_url + "|" + media_url
            yield CandidateAsset(
                asset_id=stable_asset_id(self.name, provider_id), provider=self.name,
                provider_id=provider_id, source_page_url=source_url, media_url=media_url,
                media_type="image", retrieval_timestamp=utc_now(), creator_credit=None,
                rights_statement=LEAD_RIGHTS, rights_url=None, permission_decision="pending",
                original_search_term=query, candidate_category=candidate_category,
                source_group_id=f"brave:{stable_asset_id(self.name, source_url)}", viewpoint=viewpoint,
                metadata={"title": item.get("title"), "lead_only": True},
            )


class SerpAPIImageProvider(DiscoveryProvider):
    name = "serpapi"
    requires_credentials = True

    def __init__(self, client, **options: Any):
        super().__init__(client, **options)
        self.api_key = str(options.get("api_key") or os.environ.get("SERPAPI_API_KEY") or "")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def discover(self, *, query: str, candidate_category: str, limit: int, media_types: set[str], viewpoint: str | None = None) -> Iterator[CandidateAsset]:
        if not self.available:
            raise ProviderUnavailable("SerpAPI discovery requires SERPAPI_API_KEY")
        if "image" not in media_types:
            return
        payload = self.client.get(
            "https://serpapi.com/search.json",
            params={"engine": "google_images", "q": query, "api_key": self.api_key},
        ).json()
        for item in (payload.get("images_results") or [])[:limit]:
            media_url = str(item.get("original") or "")
            source_url = str(item.get("link") or item.get("source") or "")
            if not media_url or not source_url:
                continue
            provider_id = source_url + "|" + media_url
            yield CandidateAsset(
                asset_id=stable_asset_id(self.name, provider_id), provider=self.name,
                provider_id=provider_id, source_page_url=source_url, media_url=media_url,
                media_type="image", retrieval_timestamp=utc_now(), creator_credit=None,
                rights_statement=LEAD_RIGHTS, rights_url=None, permission_decision="pending",
                original_search_term=query, candidate_category=candidate_category,
                source_group_id=f"serpapi:{stable_asset_id(self.name, source_url)}", viewpoint=viewpoint,
                width=int(item["original_width"]) if item.get("original_width") else None,
                height=int(item["original_height"]) if item.get("original_height") else None,
                metadata={"title": item.get("title"), "lead_only": True},
            )

