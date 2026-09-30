from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "datasets.yaml"


def _expand_path(value: str | Path, *, base: Path | None = None) -> Path:
    text = os.path.expandvars(str(value))
    path = Path(text).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


@dataclass(frozen=True)
class DatasetSettings:
    config_path: Path
    repo_root: Path
    data_root: Path
    dataset_name: str
    dataset_config: dict[str, Any]
    taxonomy_path: Path
    workspace_root: Path
    builds_root: Path

    @property
    def database_path(self) -> Path:
        return self.workspace_root / "metadata.sqlite3"

    @property
    def corpus_root(self) -> Path:
        return self.workspace_root / "corpus"

    @property
    def reports_root(self) -> Path:
        return self.workspace_root / "reports"

    @property
    def annotation_root(self) -> Path:
        return self.workspace_root / "annotation"

    @property
    def state_root(self) -> Path:
        return self.data_root / "state"

    def ensure_layout(self) -> None:
        for path in (
            self.data_root,
            self.workspace_root,
            self.corpus_root,
            self.reports_root,
            self.annotation_root,
            self.builds_root,
            self.state_root,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_dataset_settings(
    config_path: str | Path | None = None,
    *,
    dataset_name: str | None = None,
    data_root: str | Path | None = None,
) -> DatasetSettings:
    path = _expand_path(config_path or DEFAULT_CONFIG_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Dataset config does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    datasets = raw.get("datasets") or {}
    name = dataset_name or os.environ.get("GARUDA_DATASET_NAME") or raw.get("selected_dataset")
    if not name or name not in datasets:
        available = ", ".join(sorted(datasets)) or "none"
        raise ValueError(f"Unknown dataset {name!r}; configured datasets: {available}")

    root_value = (
        data_root
        or os.environ.get(str(raw.get("data_root_env") or "GARUDA_DATA_ROOT"))
        or raw.get("default_data_root")
        or "~/GarudaEdgeData"
    )
    root = _expand_path(root_value)
    cfg = dict(datasets[name] or {})
    taxonomy = _expand_path(cfg["taxonomy"], base=REPO_ROOT)
    workspace = _expand_path(cfg.get("workspace", f"workspaces/{name}"), base=root)
    builds = _expand_path(cfg.get("builds", f"datasets/{name}"), base=root)
    return DatasetSettings(path, REPO_ROOT, root, name, cfg, taxonomy, workspace, builds)


def resolve_selected_data_yaml(
    config_path: str | Path | None = None,
    *,
    dataset_name: str | None = None,
    version: str | None = None,
    data_root: str | Path | None = None,
    require_exists: bool = True,
) -> Path:
    """Return the selected version's absolute ``data.yaml`` path for future training.

    A successful build writes ``<GARUDA_DATA_ROOT>/state/selected_dataset.yaml``.
    Explicit arguments and environment variables take precedence over that state.
    """

    settings = load_dataset_settings(config_path, dataset_name=dataset_name, data_root=data_root)
    state_path = settings.state_root / "selected_dataset.yaml"
    state: dict[str, Any] = {}
    if state_path.exists():
        state = yaml.safe_load(state_path.read_text(encoding="utf-8")) or {}

    requested_version = (
        version
        or os.environ.get("GARUDA_DATASET_VERSION")
        or (state.get("version") if state.get("dataset_name") == settings.dataset_name else None)
    )
    if requested_version in (None, "latest"):
        candidates = sorted(
            (p for p in settings.builds_root.iterdir() if p.is_dir() and (p / "data.yaml").exists()),
            key=lambda p: p.name,
        ) if settings.builds_root.exists() else []
        if not candidates:
            result = settings.builds_root / "latest" / "data.yaml"
        else:
            result = candidates[-1] / "data.yaml"
    else:
        result = settings.builds_root / str(requested_version) / "data.yaml"

    if require_exists and not result.exists():
        raise FileNotFoundError(
            f"Selected dataset descriptor does not exist: {result}. "
            "Run `garuda-dataset build-dataset --version <version>` after reviewed annotations are imported."
        )
    return result.resolve()

