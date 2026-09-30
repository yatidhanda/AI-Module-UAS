from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Taxonomy:
    name: str
    version: str
    classes: tuple[dict[str, Any], ...]
    source_term_mappings: dict[str, str]
    search_profiles: dict[str, Any]
    path: Path

    @property
    def names(self) -> list[str]:
        return [str(item["name"]) for item in self.classes]

    @property
    def ids(self) -> set[int]:
        return {int(item["id"]) for item in self.classes}

    def candidate_for_term(self, term: str) -> str:
        return self.source_term_mappings.get(term.strip().lower(), "needs_review")


def load_taxonomy(path: str | Path) -> Taxonomy:
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    classes = tuple(raw.get("classes") or ())
    ids = [int(item["id"]) for item in classes]
    names = [str(item["name"]) for item in classes]
    if ids != list(range(len(ids))):
        raise ValueError(f"Taxonomy class IDs must be contiguous from zero: {ids}")
    if len(names) != len(set(names)):
        raise ValueError("Taxonomy class names must be unique")
    return Taxonomy(
        name=str(raw["taxonomy_name"]),
        version=str(raw["version"]),
        classes=classes,
        source_term_mappings={str(k).lower(): str(v) for k, v in (raw.get("source_term_mappings") or {}).items()},
        search_profiles=dict(raw.get("search_profiles") or {}),
        path=source,
    )

