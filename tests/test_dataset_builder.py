from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from garuda_edge.dataset_builder.annotations import export_for_annotation, import_annotations
from garuda_edge.dataset_builder.build import build_dataset, validate_dataset
from garuda_edge.dataset_builder.collection import discover_assets
from garuda_edge.dataset_builder.config import load_dataset_settings, resolve_selected_data_yaml
from garuda_edge.dataset_builder.media import download_assets, verify_assets
from garuda_edge.dataset_builder.media import extract_video_frames
from garuda_edge.dataset_builder.providers.dvids import COPYRIGHT_URL, DVIDSProvider
from garuda_edge.dataset_builder.providers.wikimedia import WikimediaCommonsProvider
from garuda_edge.dataset_builder.providers.wikimedia import permission_from_commons
from garuda_edge.dataset_builder.quality import deduplicate_assets, set_review_status
from garuda_edge.dataset_builder.store import MetadataStore
from garuda_edge.dataset_builder.taxonomy import load_taxonomy


FIXTURES = Path(__file__).parent / "fixtures"


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class FakeClient:
    def __init__(self, payload):
        self.payload = payload

    def get(self, url, **kwargs):
        return FakeResponse(self.payload)


def test_wikimedia_preserves_rights_and_quarantines_ambiguous_metadata():
    payload = json.loads((FIXTURES / "wikimedia_api_response.json").read_text(encoding="utf-8"))
    provider = WikimediaCommonsProvider(FakeClient(payload))

    items = list(
        provider.discover(
            query="aircraft aerial",
            candidate_category="fixed_wing_aircraft",
            limit=2,
            media_types={"image"},
        )
    )

    assert len(items) == 2
    assert items[0].permission_decision == "approved"
    assert items[0].rights_statement == "Public domain"
    assert items[0].creator_credit == "Fixture Creator"
    assert items[1].permission_decision == "pending"
    assert items[1].original_search_term == "aircraft aerial"
    assert permission_from_commons("CC BY-NC 4.0", "https://creativecommons.org/licenses/by-nc/4.0") == "pending"


def test_dvids_uses_documented_apis_but_requires_item_rights_review():
    search_payload = json.loads((FIXTURES / "dvids_search_response.json").read_text(encoding="utf-8"))

    class DVIDSClient:
        def get(self, url, **kwargs):
            if url.endswith("/search"):
                return FakeResponse(search_payload)
            return FakeResponse(
                {
                    "results": {
                        "id": "image:12345",
                        "title": "Mocked government image",
                        "image": "https://cdn.example.test/image.jpg",
                        "url": "https://www.dvidshub.net/image/12345/mock",
                        "credit": [{"rank": "Sgt.", "name": "Fixture Creator"}],
                        "dimensions": {"width": "1200", "height": "800"},
                    }
                }
            )

    provider = DVIDSProvider(DVIDSClient(), api_key="key-fixture")
    items = list(provider.discover(query="truck", candidate_category="truck", limit=1, media_types={"image"}))

    assert len(items) == 1
    assert items[0].permission_decision == "pending"
    assert items[0].rights_url == COPYRIGHT_URL
    assert items[0].width == 1200
    assert "Fixture Creator" in items[0].creator_credit


def test_complete_local_cvat_and_yolo_build(tmp_path: Path):
    Image = pytest.importorskip("PIL.Image")
    source = tmp_path / "source"
    source.mkdir()
    # Distinct geometric patterns make these tiny fixtures useful for hash and
    # annotation tests without carrying third-party media in the repository.
    for index in range(4):
        image = Image.new("RGB", (96, 64), color=(30 + index * 40, 50, 80))
        for offset in range(8 + index * 7, 80, 19):
            for y in range(8, 56):
                image.putpixel((min(offset, 95), y), (250, 240 - index * 20, 20 + index * 30))
        image.save(source / f"fixture-{index}.png")

    taxonomy_path = tmp_path / "taxonomy.yaml"
    taxonomy_path.write_text(
        yaml.safe_dump(
            {
                "taxonomy_name": "fixture_objects",
                "version": "1.0.0",
                "classes": [
                    {"id": 0, "name": "object_a", "description": "fixture"},
                    {"id": 1, "name": "object_b", "description": "fixture"},
                ],
                "source_term_mappings": {"fixture": "object_a"},
                "search_profiles": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "datasets.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "default_data_root": str(tmp_path / "external"),
                "selected_dataset": "fixture",
                "datasets": {
                    "fixture": {
                        "taxonomy": str(taxonomy_path),
                        "workspace": "workspaces/fixture",
                        "builds": "datasets/fixture",
                        "split": {"train": 0.5, "val": 0.25, "test": 0.25, "seed": 7},
                        "validation": {"minimum_class_instances": 1, "warn_below_class_instances": 2},
                        "deduplication": {"perceptual_hash_distance": 0},
                        "media": {"minimum_width": 32, "minimum_height": 32, "maximum_download_bytes": 1000000},
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    settings = load_dataset_settings(config_path)
    taxonomy = load_taxonomy(taxonomy_path)

    discovery = discover_assets(
        settings,
        taxonomy,
        provider_name="local",
        queries=["fixture"],
        category="object_a",
        limit=10,
        media_types={"image"},
        provider_options={
            "path": source,
            "source_name": "generated rights-cleared test fixtures",
            "creator_credit": "GarudaEdge tests",
            "rights_statement": "Generated by this test; dedicated to CC0 1.0",
            "rights_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "permission_decision": "approved",
        },
    )
    assert discovery["created"] == 4
    assert download_assets(settings, limit=10)["downloaded"] == 4
    assert verify_assets(settings)["valid"] == 4
    dedup = deduplicate_assets(settings, threshold=0)
    assert dedup["examined"] == 4

    store = MetadataStore(settings.database_path)
    assets = store.list_assets(order_by="asset_id")
    positive_ids = [str(asset["asset_id"]) for asset in assets[:3]]
    negative_id = str(assets[3]["asset_id"])
    set_review_status(settings, positive_ids, status="accepted", reviewer="fixture-reviewer")
    set_review_status(settings, [negative_id], status="negative", reviewer="fixture-reviewer")

    staged = export_for_annotation(settings, taxonomy, name="fixture-roundtrip")
    assert staged["status"] == "UNANNOTATED"
    assert staged["label_files_created"] == 0
    staging_root = Path(staged["path"])
    reviewed = tmp_path / "reviewed-cvat-export"
    shutil.copytree(staging_root, reviewed)
    labels = reviewed / "labels"
    labels.mkdir()
    for index, asset_id in enumerate(positive_ids):
        class_id = 0 if index < 2 else 1
        (labels / f"{asset_id}.txt").write_text(
            f"{class_id} 0.500000 0.500000 0.500000 0.500000\n", encoding="utf-8"
        )
    # The negative is included in staging because it was explicitly reviewed, but its
    # CVAT export must contain a deliberately empty label.
    (labels / f"{negative_id}.txt").write_text("", encoding="utf-8")
    (reviewed / "annotation_metadata.json").write_text(
        json.dumps({"items": {positive_ids[0]: {"reviewer": "cvat-user"}}}), encoding="utf-8"
    )

    imported = import_annotations(
        settings,
        taxonomy,
        source=reviewed,
        annotation_version="fixture-v1",
        reviewer="fixture-reviewer",
    )
    assert imported["success"] is True
    assert imported["validated"] == 4
    assert imported["boxes"] == 3
    assert store.get_annotation(positive_ids[0]) is not None

    built = build_dataset(settings, taxonomy, version="fixture-v1")
    assert built["valid"] is True
    assert built["assets"] == 4
    assert built["boxes"] == 3
    assert set(built["splits"]) == {"train", "val", "test"}
    validation = validate_dataset(built["path"], settings=settings, taxonomy=taxonomy)
    assert validation["valid"] is True
    assert not validation["errors"]
    selected = resolve_selected_data_yaml(config_path)
    assert selected == Path(built["data_yaml"])
    assert selected.exists()
    provenance = [json.loads(line) for line in (Path(built["path"]) / "provenance.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(item["permission_decision"] == "approved" for item in provenance)
    assert sum(item["verified_negative"] for item in provenance) == 1


def test_annotation_import_rejects_out_of_bounds_box_without_mutation(tmp_path: Path):
    # Exercise the strict parser directly through its public import behavior using the
    # smallest possible prepared SQLite record.
    Image = pytest.importorskip("PIL.Image")
    taxonomy_path = tmp_path / "taxonomy.yaml"
    taxonomy_path.write_text(
        "taxonomy_name: t\nversion: '1'\nclasses:\n  - {id: 0, name: thing}\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "datasets.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "selected_dataset": "t",
                "default_data_root": str(tmp_path / "data"),
                    "datasets": {
                        "t": {
                            "taxonomy": str(taxonomy_path),
                            "validation": {"minimum_class_instances": 1},
                            "media": {"minimum_width": 32, "minimum_height": 32},
                        }
                    },
            }
        ),
        encoding="utf-8",
    )
    settings = load_dataset_settings(config_path)
    taxonomy = load_taxonomy(taxonomy_path)
    source = tmp_path / "one.png"
    Image.new("RGB", (64, 64), "white").save(source)
    discover_assets(
        settings, taxonomy, provider_name="local", queries=["one"], category="thing", limit=1,
        media_types={"image"}, provider_options={"path": source, "rights_statement": "CC0", "permission_decision": "approved"},
    )
    download_assets(settings, limit=1)
    verify_assets(settings)
    store = MetadataStore(settings.database_path)
    asset = store.list_assets()[0]
    set_review_status(settings, [asset["asset_id"]], status="accepted")
    export = export_for_annotation(settings, taxonomy, name="bad")
    root = Path(export["path"])
    (root / f"{asset['asset_id']}.txt").write_text("0 0.95 0.5 0.2 0.2\n", encoding="utf-8")

    result = import_annotations(settings, taxonomy, source=root, annotation_version="bad-v1", reviewer="reviewer")

    assert result["success"] is False
    assert any("outside image bounds" in error for error in result["errors"])
    assert store.get_annotation(asset["asset_id"]) is None


def test_video_frames_inherit_provenance_and_source_group(tmp_path: Path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    source = tmp_path / "source"
    source.mkdir()
    video_path = source / "owned-flight.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 4.0, (160, 120))
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer unavailable")
    for index in range(12):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[:, :] = (20 + index * 10, 30, 40)
        if index >= 6:
            frame[20:100, 40:120] = (240, 240, 240)
        writer.write(frame)
    writer.release()

    taxonomy_path = tmp_path / "taxonomy.yaml"
    taxonomy_path.write_text("taxonomy_name: t\nversion: '1'\nclasses:\n  - {id: 0, name: thing}\n", encoding="utf-8")
    config_path = tmp_path / "datasets.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "selected_dataset": "t",
                "default_data_root": str(tmp_path / "data"),
                "datasets": {
                    "t": {
                        "taxonomy": str(taxonomy_path),
                        "media": {"minimum_width": 32, "minimum_height": 32, "maximum_download_bytes": 10000000},
                        "video": {
                            "frame_interval_seconds": 0.5,
                            "scene_change_threshold": 0.1,
                            "adjacent_hash_distance": 0,
                            "maximum_frames_per_video": 5,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    settings = load_dataset_settings(config_path)
    taxonomy = load_taxonomy(taxonomy_path)
    discover_assets(
        settings,
        taxonomy,
        provider_name="local",
        queries=["owned flight"],
        category="needs_review",
        limit=1,
        media_types={"video"},
        provider_options={
            "path": video_path,
            "source_name": "owned-flight-fixture",
            "rights_statement": "Owned synthetic video; CC0",
            "permission_decision": "approved",
        },
    )
    assert download_assets(settings, limit=1)["downloaded"] == 1
    assert verify_assets(settings)["valid"] == 1
    report = extract_video_frames(settings, limit_videos=1)
    assert 1 <= report["frames_created"] <= 5
    store = MetadataStore(settings.database_path)
    parent = store.list_assets(where="media_type='video'")[0]
    children = store.list_assets(where="parent_asset_id=?", params=(parent["asset_id"],))
    assert len(children) == report["frames_created"]
    assert all(child["source_group_id"] == parent["source_group_id"] for child in children)
    assert all(child["rights_statement"] == parent["rights_statement"] for child in children)
