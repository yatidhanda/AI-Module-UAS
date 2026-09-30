"""Provenance-first dataset collection and YOLO dataset assembly.

This package is deliberately separate from GarudaEdge's real-time inference path.
Optional media dependencies are imported only inside the operations that need them.
"""

from .config import DatasetSettings, load_dataset_settings, resolve_selected_data_yaml

__all__ = ["DatasetSettings", "load_dataset_settings", "resolve_selected_data_yaml"]

