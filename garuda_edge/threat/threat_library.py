from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_LIBRARY: dict[str, Any] = {
    "version": 0,
    "description": "Fallback mission/platform risk library.",
    "score_range": [0, 100],
    "levels": {
        "low": {"min": 0, "max": 100, "color": "green", "label": "Low"},
    },
    "default_object": {
        "base_score": 0,
        "category": "unknown",
        "rationale": "Threat library unavailable.",
        "aliases": [],
    },
    "object_profiles": {},
    "context_modifiers": {},
    "alerting": {"high_alert_threshold": 60, "critical_alert_threshold": 80, "max_alerts_visible": 8},
    "_library_available": False,
}


def load_threat_library(path: str | Path = "configs/threat_library.yaml") -> dict[str, Any]:
    config_path = _resolve_path(path)
    if not config_path.exists():
        return deepcopy(DEFAULT_LIBRARY)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return deepcopy(DEFAULT_LIBRARY)
    if not isinstance(payload, dict):
        return deepcopy(DEFAULT_LIBRARY)
    library = deepcopy(DEFAULT_LIBRARY)
    library.update(payload)
    library["_library_available"] = True
    return library


def resolve_object_profile(class_name: str, library: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize(class_name)
    profiles = library.get("object_profiles") if isinstance(library.get("object_profiles"), dict) else {}
    for key, profile in profiles.items():
        if normalized == _normalize(key):
            return _with_profile_name(key, profile, library)
        aliases = profile.get("aliases") if isinstance(profile, dict) else []
        if any(normalized == _normalize(alias) for alias in aliases or []):
            return _with_profile_name(key, profile, library)

    default_profile = library.get("default_object") if isinstance(library.get("default_object"), dict) else {}
    profile = deepcopy(default_profile)
    profile.setdefault("category", "unknown")
    profile.setdefault("base_score", 0 if not library.get("_library_available", True) else 10)
    profile.setdefault("rationale", "Unknown object. Requires mission context before assigning higher risk.")
    profile["name"] = normalized or "unknown"
    profile["matched_default"] = True
    return profile


def _with_profile_name(key: str, profile: dict[str, Any], library: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(library.get("default_object") if isinstance(library.get("default_object"), dict) else {})
    merged.update(profile if isinstance(profile, dict) else {})
    merged.setdefault("category", "unknown")
    merged.setdefault("base_score", 10)
    merged.setdefault("rationale", "Object profile matched. Review depends on mission context.")
    merged["name"] = key
    merged["matched_default"] = False
    return merged


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved
