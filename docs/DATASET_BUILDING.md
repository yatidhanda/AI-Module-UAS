# GarudaEdge Part A dataset pipeline

This package collects permitted public or locally licensed media, preserves provenance,
stages images for human review and CVAT annotation, imports reviewed bounding boxes, and
builds an immutable YOLO detection dataset. It does not train YOLO or Gemma and is not
used by the real-time inference, tracking, geolocation, policy, or operator-action paths.

## Clean-clone setup on macOS

```bash
git clone https://github.com/yatidhanda/AI-Module-UAS.git
cd AI-Module-UAS
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dataset]"
garuda-dataset --help
```

The edge runtime does not receive Pillow or the collection-specific media dependencies
unless the `dataset` extra is installed.

Large files never belong in the checkout. The default is `~/GarudaEdgeData`; override it
for an external SSD before any command:

```bash
export GARUDA_DATA_ROOT="/Volumes/GarudaData/GarudaEdgeData"
mkdir -p "$GARUDA_DATA_ROOT"
```

`configs/datasets.yaml` is the small tracked catalog. It names the taxonomy, workspace,
build directory, split seed, media limits, and quality thresholds. Every relative data
path is resolved below `GARUDA_DATA_ROOT`. The workspace contains `metadata.sqlite3`,
the immutable downloaded corpus, quarantined items, review reports, and annotation
staging exports. Built versions are under `datasets/garuda_objects/<version>`.

Do not put API keys in the YAML or Git. Optional providers read:

```bash
export DVIDS_API_KEY="key-..."
export BRAVE_SEARCH_API_KEY="..."
export SERPAPI_API_KEY="..."
```

Wikimedia Commons needs no key. [DVIDS Search API](https://api.dvidshub.net/docs/search_api)
currently requires a registered API key and its
[rights notice](https://www.dvidshub.net/about/copyright) says that some hosted material has third-party rights. Consequently, DVIDS
records are discovery leads with `permission_decision=pending` until a human records an
item-specific decision. Brave and SerpAPI results are also leads, never license evidence.

## Taxonomy and data contract

`configs/taxonomy/garuda_objects_v1.yaml` owns a versioned set of broad, visually
supportable classes. It retains mappings from discovery terms to candidate categories,
but those mappings never create annotations. `unknown` and `needs_review` are workflow
states, not training classes. The existing runtime detector is unchanged: it will keep
using the class names supplied by its currently loaded model. A future model trained on a
built dataset will supply this taxonomy through the same detector interface.

Each SQLite and JSONL asset record retains its stable ID, provider ID, source page and
media URLs, retrieval time, creator/credit, exact rights statement and URL, permission
decision, every search query, candidate category, review state, SHA-256, dHash, media
dimensions or video metadata, source group, derived-frame IDs, and failures. Failed and
rejected records remain in metadata. Automated quality steps mark state; they never
delete originals.

## A bounded 100–200 candidate pilot

Run discovery first. Search-profile terms intentionally cover aerial/overhead views,
varied backgrounds, broad objects, and civilian lookalikes:

```bash
garuda-dataset discover \
  --provider wikimedia \
  --profile road_and_civilian_lookalikes \
  --limit 120 \
  --dry-run

garuda-dataset discover \
  --provider wikimedia \
  --profile road_and_civilian_lookalikes \
  --limit 120

garuda-dataset download --limit 120 --dry-run
garuda-dataset download --limit 120
garuda-dataset verify --limit 120
garuda-dataset deduplicate --dry-run
garuda-dataset deduplicate
garuda-dataset stats
```

Use `--include-pending-rights` only when a human needs a local quarantined copy to resolve
rights; it cannot enter a build while pending. Every request has timeouts, a descriptive
User-Agent, per-host pacing, bounded retries/backoff, pagination limits, and a maximum
download size. Partial network downloads use `.part` files and HTTP Range requests when
the server supports them. Official API media URLs are used directly. Search-engine leads
are additionally subject to a fail-closed robots check. No code bypasses authentication,
CAPTCHAs, paywalls, or access controls.

For owned footage or a separately licensed research dataset, provide its exact rights:

```bash
garuda-dataset discover \
  --provider local \
  --local-path /Volumes/Footage/approved-pilot \
  --source-name owned-flight-2026-09 \
  --creator-credit "Your organization" \
  --rights-statement "Owned footage; approved for model-development use by ..." \
  --rights-url "file:///Volumes/Footage/licenses/approval.txt" \
  --permission-decision approved \
  --category needs_review \
  --media-type image \
  --media-type video \
  --limit 200

garuda-dataset download --limit 200
garuda-dataset verify
garuda-dataset extract-frames --limit-videos 10
```

Video provenance stays on the parent and every extracted frame. Interval and scene-change
sampling are both enabled; adjacent frames below the configured dHash distance are not
added. Every frame from one video uses the same `source_group_id`. Video rights are
reviewed on the parent source and inherited by frames, not inferred from image rights.

## Rights and visual review

Generate the local contact sheet, open the printed HTML path, and record decisions:

```bash
garuda-dataset review-report --limit 200
open "$GARUDA_DATA_ROOT/workspaces/garuda_objects/reports/review.html"

garuda-dataset review \
  --asset-id wikimedia-EXAMPLE \
  --status accepted \
  --reviewer "reviewer@example.org"

garuda-dataset review \
  --asset-id wikimedia-NEGATIVE \
  --status negative \
  --reason "Verified civilian lookalike; no taxonomy object present" \
  --reviewer "reviewer@example.org"

garuda-dataset review \
  --asset-id dvids-EXAMPLE \
  --status accepted \
  --permission-decision approved \
  --rights-statement "Exact item-level rights statement copied from the source" \
  --rights-url "https://source.example/item/rights" \
  --reviewer "reviewer@example.org"
```

Use `rejected` for unsuitable images and `ambiguous` when visual content cannot support a
class. A caption, search ranker, or automated classifier may prioritize review, but it may
not establish a bounding box or class.

## The required human CVAT step

```bash
garuda-dataset export-for-annotation --name pilot-2026-09 --limit 200
```

The command creates a directory and ZIP below
`$GARUDA_DATA_ROOT/workspaces/garuda_objects/annotation/staging/`. It includes images,
`obj.names`, `obj.data`, `train.txt`, `taxonomy.yaml`, provenance, and a prominent
`README_UNANNOTATED.md`. It creates **zero label files**. In YOLO, an absent or empty label
means “no objects”; generating empty files for unreviewed positives would silently poison
the dataset.

In CVAT:

1. Create a task with labels in the exact `obj.names` order and upload the staging images.
2. Follow `README_UNANNOTATED.md`: tight visible boxes; every relevant visible object;
   visually supportable broad classes only; consistent occlusion/truncation; unresolved
   ambiguity returned for review.
3. For images already marked `negative`, confirm that no taxonomy object is present and
   leave the image intentionally empty. Do not turn an unreviewed empty image into a
   negative.
4. Complete CVAT review, then export **Ultralytics YOLO Detection**. Preserve the images
   and label files in the export.

Import only the reviewed export:

```bash
garuda-dataset import-annotations \
  --input ~/Downloads/cvat-pilot-yolo.zip \
  --annotation-version pilot-2026-09-r1 \
  --reviewer "reviewer@example.org" \
  --dry-run

garuda-dataset import-annotations \
  --input ~/Downloads/cvat-pilot-yolo.zip \
  --annotation-version pilot-2026-09-r1 \
  --reviewer "reviewer@example.org"
```

Import is all-or-nothing. It matches the stable asset ID in each filename, verifies image
SHA-256 and dimensions, requires an exact taxonomy match and a label for every image,
rejects invalid class IDs and malformed/out-of-bounds/zero-area boxes, refuses empty
positive labels, and permits empty labels only for previously reviewed negatives.
`annotation_metadata.json`, when present, may override reviewer/version per stable ID.

## Build, validate, and resolve `data.yaml`

```bash
garuda-dataset build-dataset --version pilot-2026-09-v1 --dry-run
garuda-dataset build-dataset --version pilot-2026-09-v1
garuda-dataset validate-dataset \
  --path "$GARUDA_DATA_ROOT/datasets/garuda_objects/pilot-2026-09-v1"
garuda-dataset data-path
```

A version is immutable; rebuilding the same name fails instead of overwriting it. Splits
are deterministic from the configured seed and grouped by source group and near-duplicate
cluster, so a video's frames cannot cross splits. A build fails on missing images/labels,
bad IDs or boxes, content/group/cluster leakage, unresolved rights, unreviewed items,
unvalidated annotations, empty required splits, and classes below
`minimum_class_instances`. The tracked production catalog requires 20 validated boxes per
class (and warns below 100); a 100–200 candidate collection pilot will often remain an
annotation/review exercise rather than qualify as a training-ready release.

Every version contains:

```text
images/{train,val,test}/
labels/{train,val,test}/
data.yaml
provenance.jsonl
licenses.csv
DATASET_CARD.md
taxonomy.yaml
taxonomy_version.txt
stats.json
build_config.yaml
validation_report.json
```

Later Python training code can resolve the same selected version without hard-coded
paths. This is a path lookup only; Part A does not train a model:

```python
from garuda_edge.dataset_builder import resolve_selected_data_yaml

data_yaml = resolve_selected_data_yaml()
```

The successful build writes the selection to
`$GARUDA_DATA_ROOT/state/selected_dataset.yaml`. `GARUDA_DATASET_NAME` and
`GARUDA_DATASET_VERSION` can override selection.

## Disk use and backups

Budget for original media plus quarantined media, extracted frames, annotation staging,
and each materialized build. A 150-image pilot is commonly hundreds of MB; video can make
it tens of GB. Builds use hard links when corpus and build are on the same filesystem, so
they usually add little image-byte overhead; cross-filesystem builds fall back to copies.

Back up both bytes and metadata. An image-only backup cannot reconstruct rights decisions,
review state, stable identities, or CVAT history.

```bash
rsync -a --info=progress2 "$GARUDA_DATA_ROOT/" /Volumes/Backup/GarudaEdgeData/
sqlite3 "$GARUDA_DATA_ROOT/workspaces/garuda_objects/metadata.sqlite3" \
  ".backup '/Volumes/Backup/garuda-metadata.sqlite3'"
```

Keep the generated `candidate_assets.jsonl`, `candidate_assets.csv`, validation reports,
and CVAT exports with the SQLite backup. Provider terms may require periodic rights or
publication-status revalidation; the stored URLs and retrieval timestamps make that
possible.
