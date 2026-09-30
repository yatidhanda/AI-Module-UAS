from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import os
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .collection import make_client, write_workspace_manifests
from .config import DatasetSettings
from .http import require_requests
from .models import utc_now
from .providers.base import stable_asset_id
from .store import MetadataStore


IMAGE_MIME_PREFIX = "image/"
VIDEO_MIME_PREFIX = "video/"
MIME_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/tiff": ".tif",
    "image/bmp": ".bmp",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "video/x-msvideo": ".avi",
}


def _require_pillow():
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Image verification requires Pillow. Install `python -m pip install -e '.[dataset]'`."
        ) from exc
    return Image, UnidentifiedImageError


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_dhash(path: Path) -> str:
    Image, _ = _require_pillow()
    with Image.open(path) as image:
        resized = image.convert("L").resize((9, 8))
        getter = getattr(resized, "get_flattened_data", resized.getdata)
        pixels = list(getter())
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | int(pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return f"{value:016x}"


def hash_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _extension(asset: dict[str, Any], content_type: str | None = None) -> str:
    normalized = (content_type or "").split(";", 1)[0].lower()
    if normalized in MIME_EXTENSIONS:
        return MIME_EXTENSIONS[normalized]
    url_suffix = Path(urlparse(str(asset["media_url"])).path).suffix.lower()
    if 1 < len(url_suffix) <= 6:
        return url_suffix
    guessed = mimetypes.guess_extension(str(asset.get("mime_type") or ""))
    return guessed or (".mp4" if asset["media_type"] == "video" else ".jpg")


def download_assets(
    settings: DatasetSettings,
    *,
    limit: int,
    include_pending_rights: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    settings.ensure_layout()
    store = MetadataStore(settings.database_path)
    where = "download_status IN ('discovered','failed')"
    params: list[Any] = []
    if not include_pending_rights:
        where += " AND permission_decision='approved'"
    rows = store.list_assets(where=where, params=params, limit=limit)
    report = {"selected": len(rows), "downloaded": 0, "quarantined": 0, "failed": 0, "skipped": 0, "dry_run": dry_run}
    if dry_run:
        report["assets"] = [row["asset_id"] for row in rows]
        return report
    media_cfg = settings.dataset_config.get("media") or {}
    max_bytes = int(media_cfg.get("maximum_download_bytes", 500 * 1024 * 1024))
    for asset in rows:
        try:
            destination_root = settings.corpus_root if asset["permission_decision"] == "approved" else settings.workspace_root / "quarantine"
            destination_root = destination_root / str(asset["media_type"])
            destination_root.mkdir(parents=True, exist_ok=True)
            if str(asset["media_url"]).startswith("file://"):
                source = Path(unquote(urlparse(str(asset["media_url"])).path))
                if not source.exists():
                    raise FileNotFoundError(f"Local source no longer exists: {source}")
                if source.stat().st_size > max_bytes:
                    raise ValueError(f"local file exceeds maximum_download_bytes={max_bytes}")
                suffix = _extension(asset)
                target = destination_root / f"{asset['asset_id']}{suffix}"
                shutil.copy2(source, target)
                content_type = str(asset.get("mime_type") or mimetypes.guess_type(source.name)[0] or "")
            else:
                target, content_type = _stream_download(settings, asset, destination_root, max_bytes)
            status = "downloaded" if asset["permission_decision"] == "approved" else "quarantined"
            store.update_asset(
                str(asset["asset_id"]),
                download_status=status,
                local_path=str(target.resolve()),
                byte_size=target.stat().st_size,
                mime_type=content_type.split(";", 1)[0].lower() or asset.get("mime_type"),
                failure_reason=None,
            )
            store.record_event("downloaded", asset_id=str(asset["asset_id"]), details={"path": str(target), "status": status})
            report[status] += 1
        except Exception as exc:
            store.update_asset(str(asset["asset_id"]), download_status="failed", failure_reason=str(exc))
            store.record_event("download_failed", asset_id=str(asset["asset_id"]), reason=str(exc))
            report["failed"] += 1
    write_workspace_manifests(settings, store)
    return report


def _stream_download(
    settings: DatasetSettings,
    asset: dict[str, Any],
    destination_root: Path,
    max_bytes: int,
) -> tuple[Path, str]:
    requests = require_requests()
    client = make_client(settings)
    url = str(asset["media_url"])
    # Direct media URLs returned by documented provider APIs are intentionally fetched;
    # arbitrary search-engine leads are checked against robots.txt and fail closed.
    if asset["provider"] in {"brave", "serpapi"} and not client.allowed_by_robots(url, fail_closed=True):
        raise PermissionError(f"robots.txt did not explicitly permit lead media URL: {url}")
    last_error: Exception | None = None
    for attempt in range(client.retries + 1):
        try:
            return _stream_download_once(client, requests, asset, destination_root, max_bytes)
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= client.retries:
                break
            client._wait_for_retry(None, attempt)
    raise RuntimeError(f"media download failed after {client.retries + 1} attempts for {url}: {last_error}")


def _stream_download_once(
    client,
    requests,
    asset: dict[str, Any],
    destination_root: Path,
    max_bytes: int,
) -> tuple[Path, str]:
    url = str(asset["media_url"])
    initial_suffix = _extension(asset)
    target = destination_root / f"{asset['asset_id']}{initial_suffix}"
    partial = target.with_suffix(target.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": client.user_agent}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    client._rate_limit(url)
    with requests.get(url, headers=headers, timeout=client.timeout_seconds, stream=True) as response:
        if response.status_code == 416 and existing:
            partial.replace(target)
            return target, str(asset.get("mime_type") or "")
        response.raise_for_status()
        if existing and response.status_code != 206:
            existing = 0
            partial.unlink(missing_ok=True)
        content_length = int(response.headers.get("Content-Length") or 0)
        if content_length and existing + content_length > max_bytes:
            raise ValueError(f"server content length exceeds maximum_download_bytes={max_bytes}")
        content_type = str(response.headers.get("Content-Type") or asset.get("mime_type") or "")
        suffix = _extension(asset, content_type)
        if suffix != initial_suffix:
            target = target.with_suffix(suffix)
            partial = target.with_suffix(target.suffix + ".part")
            existing = partial.stat().st_size if partial.exists() else 0
        mode = "ab" if existing else "wb"
        size = existing
        with partial.open(mode) as fp:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError(f"download exceeded maximum_download_bytes={max_bytes}")
                fp.write(chunk)
        os.replace(partial, target)
    return target, content_type


def verify_assets(settings: DatasetSettings, *, limit: int | None = None) -> dict[str, Any]:
    settings.ensure_layout()
    store = MetadataStore(settings.database_path)
    rows = store.list_assets(
        where="download_status IN ('downloaded','quarantined','local') AND local_path IS NOT NULL",
        limit=limit,
    )
    media_cfg = settings.dataset_config.get("media") or {}
    min_width = int(media_cfg.get("minimum_width", 320))
    min_height = int(media_cfg.get("minimum_height", 240))
    report = {"examined": len(rows), "valid": 0, "invalid": 0, "images": 0, "videos": 0, "failures": []}
    for asset in rows:
        path = Path(str(asset["local_path"]))
        try:
            if not path.exists():
                raise FileNotFoundError(f"missing downloaded file: {path}")
            digest = sha256_file(path)
            if asset["media_type"] == "image":
                metadata = _verify_image(path, min_width, min_height)
                metadata["perceptual_hash"] = perceptual_dhash(path)
                report["images"] += 1
            elif asset["media_type"] == "video":
                metadata = _verify_video(path, min_width, min_height)
                report["videos"] += 1
            else:
                raise ValueError(f"unsupported media_type={asset['media_type']!r}")
            store.update_asset(str(asset["asset_id"]), sha256=digest, failure_reason=None, **metadata)
            store.record_event("verified", asset_id=str(asset["asset_id"]), details=metadata)
            report["valid"] += 1
        except Exception as exc:
            store.update_asset(str(asset["asset_id"]), download_status="failed", failure_reason=f"verification: {exc}")
            store.record_event("verification_failed", asset_id=str(asset["asset_id"]), reason=str(exc))
            report["failures"].append({"asset_id": asset["asset_id"], "reason": str(exc)})
            report["invalid"] += 1
    write_workspace_manifests(settings, store)
    report_path = settings.reports_root / "verification_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def _verify_image(path: Path, min_width: int, min_height: int) -> dict[str, Any]:
    Image, UnidentifiedImageError = _require_pillow()
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            image.load()
            mime = Image.MIME.get(image.format, "")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"image decode failed: {exc}") from exc
    if width < min_width or height < min_height:
        raise ValueError(f"image dimensions {width}x{height} below minimum {min_width}x{min_height}")
    if not mime.startswith(IMAGE_MIME_PREFIX):
        raise ValueError(f"decoded file is not an image (MIME {mime!r})")
    return {"width": width, "height": height, "mime_type": mime}


def _verify_video(path: Path, min_width: int, min_height: int) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Video verification requires OpenCV from the dataset optional dependencies") from exc
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("video decoder could not open file")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, _ = capture.read()
        if not ok:
            raise ValueError("video has no decodable first frame")
    finally:
        capture.release()
    if width < min_width or height < min_height:
        raise ValueError(f"video dimensions {width}x{height} below minimum {min_width}x{min_height}")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"invalid video frame rate: {fps}")
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration_seconds": frame_count / fps if frame_count else None,
        "mime_type": mimetypes.guess_type(path.name)[0] or "video/mp4",
    }


def extract_video_frames(settings: DatasetSettings, *, limit_videos: int | None = None) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Frame extraction requires OpenCV") from exc
    store = MetadataStore(settings.database_path)
    videos = store.list_assets(
        where="media_type='video' AND download_status IN ('downloaded','quarantined') AND sha256 IS NOT NULL",
        limit=limit_videos,
    )
    cfg = settings.dataset_config.get("video") or {}
    interval_s = float(cfg.get("frame_interval_seconds", 5.0))
    scene_threshold = float(cfg.get("scene_change_threshold", 0.30))
    adjacent_threshold = int(cfg.get("adjacent_hash_distance", 4))
    max_frames = int(cfg.get("maximum_frames_per_video", 200))
    report = {"videos": len(videos), "frames_created": 0, "near_adjacent_skipped": 0, "failures": []}
    for video in videos:
        capture = cv2.VideoCapture(str(video["local_path"]))
        previous_scene = previous_kept_hash = None
        output_dir = settings.corpus_root / "frames" / str(video["asset_id"])
        output_dir.mkdir(parents=True, exist_ok=True)
        created = 0
        frame_index = 0
        try:
            fps = float(video.get("fps") or capture.get(cv2.CAP_PROP_FPS) or 0)
            if fps <= 0:
                raise ValueError("missing valid fps")
            interval_frames = max(1, round(interval_s * fps))
            while created < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64))
                scene_score = 0.0
                if previous_scene is not None:
                    scene_score = float(cv2.absdiff(small, previous_scene).mean() / 255.0)
                due = frame_index == 0 or frame_index % interval_frames == 0 or scene_score >= scene_threshold
                previous_scene = small
                if not due:
                    frame_index += 1
                    continue
                time_s = frame_index / fps
                child_provider_id = f"{video['provider_id']}#frame={frame_index}"
                child_id = stable_asset_id(str(video["provider"]), child_provider_id)
                path = output_dir / f"{child_id}.jpg"
                if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                    raise ValueError(f"could not write extracted frame {frame_index}")
                digest = sha256_file(path)
                phash = perceptual_dhash(path)
                if previous_kept_hash and hash_distance(previous_kept_hash, phash) <= adjacent_threshold:
                    path.unlink(missing_ok=True)
                    report["near_adjacent_skipped"] += 1
                    frame_index += 1
                    continue
                previous_kept_hash = phash
                values = {
                    "asset_id": child_id,
                    "provider": video["provider"],
                    "provider_id": child_provider_id,
                    "source_page_url": video["source_page_url"],
                    "media_url": f"{video['media_url']}#t={time_s:.3f}",
                    "media_type": "image",
                    "retrieval_timestamp": utc_now(),
                    "creator_credit": video["creator_credit"],
                    "rights_statement": video["rights_statement"],
                    "rights_url": video["rights_url"],
                    "permission_decision": video["permission_decision"],
                    "original_search_term": video["original_search_term"],
                    "candidate_category": video["candidate_category"],
                    "review_status": "pending",
                    "download_status": "downloaded" if video["permission_decision"] == "approved" else "quarantined",
                    "sha256": digest,
                    "perceptual_hash": phash,
                    "mime_type": "image/jpeg",
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "byte_size": path.stat().st_size,
                    "local_path": str(path.resolve()),
                    "source_group_id": video["source_group_id"],
                    "parent_asset_id": video["asset_id"],
                    "frame_time_seconds": time_s,
                    "viewpoint": video["viewpoint"],
                    "metadata_json": {"frame_index": frame_index, "scene_change_score": scene_score},
                }
                if store.add_derived_asset(values):
                    created += 1
                    report["frames_created"] += 1
                else:
                    path.unlink(missing_ok=True)
                frame_index += 1
        except Exception as exc:
            report["failures"].append({"asset_id": video["asset_id"], "reason": str(exc)})
            store.record_event("frame_extraction_failed", asset_id=str(video["asset_id"]), reason=str(exc))
        finally:
            capture.release()
    write_workspace_manifests(settings, store)
    path = settings.reports_root / "frame_extraction_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(path)
    return report
