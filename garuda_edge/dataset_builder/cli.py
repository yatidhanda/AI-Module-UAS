from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .annotations import export_for_annotation, import_annotations
from .build import DatasetBuildError, build_dataset, validate_dataset
from .collection import PROVIDERS, discover_assets, profile_queries
from .config import load_dataset_settings, resolve_selected_data_yaml
from .media import download_assets, extract_video_frames, verify_assets
from .quality import dataset_stats, deduplicate_assets, generate_review_report, set_review_status
from .store import MetadataStore
from .taxonomy import load_taxonomy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="garuda-dataset",
        description="Collect, review, annotate, and build provenance-validated GarudaEdge YOLO datasets.",
    )
    parser.add_argument("--config", type=Path, help="Dataset config (default: configs/datasets.yaml)")
    parser.add_argument("--dataset", help="Configured dataset name")
    parser.add_argument("--data-root", type=Path, help="Override GARUDA_DATA_ROOT")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover", help="Discover candidates from a provider and preserve source metadata.")
    discover.add_argument("--provider", required=True, choices=sorted(PROVIDERS))
    discover.add_argument("--query", action="append", default=[], help="Search term; repeat for multiple terms.")
    discover.add_argument("--profile", help="Named search profile from the taxonomy.")
    discover.add_argument("--category", help="Candidate category only; this never becomes a ground-truth label.")
    discover.add_argument("--viewpoint", help="Viewpoint metadata, e.g. aerial or overhead.")
    discover.add_argument("--media-type", action="append", choices=["image", "video"], default=[])
    discover.add_argument("--limit", type=_positive_int, default=100, help="Maximum candidates across all queries.")
    discover.add_argument("--dry-run", action="store_true", help="Query provider but do not write metadata.")
    discover.add_argument("--local-path", type=Path)
    discover.add_argument("--source-name")
    discover.add_argument("--source-group-id")
    discover.add_argument("--creator-credit")
    discover.add_argument("--rights-statement")
    discover.add_argument("--rights-url")
    discover.add_argument("--permission-decision", choices=["pending", "approved", "rejected"], default="pending")

    download = subparsers.add_parser("download", help="Download or copy discovered media with byte limits and resume support.")
    download.add_argument("--limit", type=_positive_int, default=100)
    download.add_argument("--include-pending-rights", action="store_true", help="Download unresolved items into quarantine.")
    download.add_argument("--dry-run", action="store_true")

    verify = subparsers.add_parser("verify", help="Validate MIME/decode/dimensions and compute content hashes.")
    verify.add_argument("--limit", type=_positive_int)

    frames = subparsers.add_parser("extract-frames", help="Extract interval and scene-change frames from permitted videos.")
    frames.add_argument("--limit-videos", type=_positive_int)

    dedup = subparsers.add_parser("deduplicate", help="Cluster exact and perceptually near-duplicate images without deleting originals.")
    dedup.add_argument("--threshold", type=_nonnegative_int, help="Maximum dHash Hamming distance.")
    dedup.add_argument("--dry-run", action="store_true")

    report = subparsers.add_parser("review-report", help="Generate an HTML contact sheet and CSV review queue.")
    report.add_argument("--limit", type=_positive_int)

    review = subparsers.add_parser("review", help="Record a human accept/reject/ambiguous/negative decision.")
    review.add_argument("--asset-id", action="append", required=True, help="Stable asset ID; repeat as needed.")
    review.add_argument("--status", required=True, choices=["accepted", "rejected", "ambiguous", "negative", "pending"])
    review.add_argument("--reason")
    review.add_argument("--reviewer")
    review.add_argument("--permission-decision", choices=["pending", "approved", "rejected"])
    review.add_argument("--rights-statement", help="Exact item rights text; required when rights are approved here.")
    review.add_argument("--rights-url")
    review.add_argument("--dry-run", action="store_true")

    export = subparsers.add_parser("export-for-annotation", help="Create a CVAT-ready image ZIP explicitly marked UNANNOTATED.")
    export.add_argument("--name", required=True)
    export.add_argument("--limit", type=_positive_int)
    export.add_argument("--dry-run", action="store_true")

    annotation_import = subparsers.add_parser("import-annotations", help="Validate and import reviewed CVAT Ultralytics YOLO Detection export.")
    annotation_import.add_argument("--input", type=Path, required=True)
    annotation_import.add_argument("--annotation-version", required=True)
    annotation_import.add_argument("--reviewer")
    annotation_import.add_argument("--dry-run", action="store_true")

    build = subparsers.add_parser("build-dataset", help="Build an immutable, versioned, validated YOLO dataset.")
    build.add_argument("--version", required=True)
    build.add_argument("--dry-run", action="store_true")

    validate = subparsers.add_parser("validate-dataset", help="Revalidate a built dataset and rewrite validation_report.json.")
    validate.add_argument("--path", type=Path, help="Dataset version root; defaults to the selected build.")

    subparsers.add_parser("stats", help="Show collection and readiness counts and write stats.json.")
    subparsers.add_parser("data-path", help="Print the selected version's data.yaml path for future training code.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = load_dataset_settings(args.config, dataset_name=args.dataset, data_root=args.data_root)
        taxonomy = load_taxonomy(settings.taxonomy_path)
        result: dict[str, Any] | str
        if args.command == "discover":
            queries = list(args.query)
            viewpoint = args.viewpoint
            if args.profile:
                profile_terms, profile_viewpoint = profile_queries(taxonomy, args.profile)
                queries.extend(profile_terms)
                viewpoint = viewpoint or profile_viewpoint
            if args.provider == "local" and not queries:
                queries = [args.source_name or "local import"]
            if not queries:
                parser.error("discover requires --query or --profile")
            if args.category and args.category not in {*taxonomy.names, "unknown", "needs_review"}:
                raise ValueError(f"Unknown candidate category {args.category!r}")
            options = {
                "path": args.local_path,
                "source_name": args.source_name,
                "source_group_id": args.source_group_id,
                "creator_credit": args.creator_credit,
                "rights_statement": args.rights_statement,
                "rights_url": args.rights_url,
                "permission_decision": args.permission_decision,
            }
            result = discover_assets(
                settings,
                taxonomy,
                provider_name=args.provider,
                queries=queries,
                category=args.category,
                limit=args.limit,
                media_types=set(args.media_type or ["image"]),
                viewpoint=viewpoint,
                dry_run=args.dry_run,
                provider_options=options,
            )
        elif args.command == "download":
            result = download_assets(settings, limit=args.limit, include_pending_rights=args.include_pending_rights, dry_run=args.dry_run)
        elif args.command == "verify":
            result = verify_assets(settings, limit=args.limit)
        elif args.command == "extract-frames":
            result = extract_video_frames(settings, limit_videos=args.limit_videos)
        elif args.command == "deduplicate":
            result = deduplicate_assets(settings, threshold=args.threshold, dry_run=args.dry_run)
        elif args.command == "review-report":
            result = generate_review_report(settings, limit=args.limit)
        elif args.command == "review":
            if args.dry_run:
                store = MetadataStore(settings.database_path)
                missing = [asset_id for asset_id in args.asset_id if store.get_asset(asset_id) is None]
                if missing:
                    raise KeyError(f"Unknown asset IDs: {missing}")
                result = {"dry_run": True, "assets": args.asset_id, "status": args.status}
            else:
                result = set_review_status(
                    settings,
                    args.asset_id,
                    status=args.status,
                    reason=args.reason,
                    permission_decision=args.permission_decision,
                    rights_statement=args.rights_statement,
                    rights_url=args.rights_url,
                    reviewer=args.reviewer,
                )
        elif args.command == "export-for-annotation":
            result = export_for_annotation(settings, taxonomy, name=args.name, limit=args.limit, dry_run=args.dry_run)
        elif args.command == "import-annotations":
            result = import_annotations(
                settings,
                taxonomy,
                source=args.input,
                annotation_version=args.annotation_version,
                reviewer=args.reviewer,
                dry_run=args.dry_run,
            )
            _print(result)
            return 0 if result.get("success") else 2
        elif args.command == "build-dataset":
            result = build_dataset(settings, taxonomy, version=args.version, dry_run=args.dry_run)
        elif args.command == "validate-dataset":
            root = args.path
            if root is None:
                root = resolve_selected_data_yaml(
                    args.config,
                    dataset_name=args.dataset,
                    data_root=args.data_root,
                ).parent
            result = validate_dataset(root, settings=settings, taxonomy=taxonomy, write_report=True)
            _print(result)
            return 0 if result.get("valid") else 2
        elif args.command == "stats":
            result = dataset_stats(settings)
        elif args.command == "data-path":
            result = str(resolve_selected_data_yaml(args.config, dataset_name=args.dataset, data_root=args.data_root))
        else:  # pragma: no cover
            parser.error(f"Unhandled command: {args.command}")
        _print(result)
        return 0
    except (DatasetBuildError, FileNotFoundError, FileExistsError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _print(result: dict[str, Any] | str) -> None:
    if isinstance(result, str):
        print(result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

