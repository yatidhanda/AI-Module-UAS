from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from ..http import PoliteHTTPClient
from ..models import CandidateAsset


class ProviderUnavailable(RuntimeError):
    pass


def stable_asset_id(provider: str, provider_id: str) -> str:
    digest = hashlib.sha256(f"{provider}:{provider_id}".encode("utf-8")).hexdigest()[:24]
    return f"{provider}-{digest}"


class DiscoveryProvider(ABC):
    name = "base"
    requires_credentials = False

    def __init__(self, client: PoliteHTTPClient, **options: Any):
        self.client = client
        self.options = options

    @property
    def available(self) -> bool:
        return True

    @abstractmethod
    def discover(
        self,
        *,
        query: str,
        candidate_category: str,
        limit: int,
        media_types: set[str],
        viewpoint: str | None = None,
    ) -> Iterator[CandidateAsset]:
        raise NotImplementedError

