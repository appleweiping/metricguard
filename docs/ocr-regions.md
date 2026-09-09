# Source-bound OCR regions

This development API lives in `metricguard.ocr_regions`. It adds **pixel geometry
and cropping**, not a learned page detector or a new recognizer. Use the current
`feat/whole-repository-alignment` checkout; these APIs are not a claim about an
already published release. The [region pipeline](ocr-region-pipeline.md) applies
a configured recognizer and assembles source-bound results.

## A small, explicit contract

`PixelBox(left, top, right, bottom)` uses integer, half-open coordinates in the
already decoded RGB source raster. Bounds must be positive-area and in the
source page. Booleans and floating-point pixel coordinates are rejected.
`OcrRegion(index, box)` identifies a position in a plan, **not** a new source
page. A crop retains the original source page index, file SHA-256, and source
type. It has its own dimensions, RGB pixel SHA-256, and versioned crop renderer
identity; the original full-page identity is retained separately.

`OcrRegionPlan` binds all of:

- The full `OcrPageInfo`: file and dimension-aware pixel digests, source page
  index/type, dimensions, and renderer.
- Ordered region boxes and contiguous indices, direction, method, configuration,
  and declared outcome.
- Explicit permission to omit pixels and the computed covered/uncovered counts.

Its digest is the SHA-256 of versioned, canonical JSON. This detects accidental
identity changes; it is not a signature or proof that an arbitrary author is
trusted. `to_dict()` includes the digest and derived pixel counts;
`OcrRegionPlan.from_dict()` checks a closed schema, all nested geometry, exact
derived counts, and the digest. It rejects unknown fields and boolean/numeric
substitutions in counters. Complete plan JSON is capped at 1 MiB. The caller
loading a JSON file must also bound the file and reject duplicate JSON keys;
`from_dict()` cannot recover duplicate keys already discarded by a JSON parser.

## Explicit selection and omissions

Pass a list or tuple of `PixelBox` values to `explicit_region_plan`. The function
does not silently sort, merge, clip, fill gaps, or discard boxes. Rectangles may
touch, but must not overlap. The declared **column-major convention** is:

- `ltr`: increasing left edge, then increasing top edge.
- `rtl`: decreasing right edge, then increasing top edge.

Remaining ties use the opposite horizontal edge, then bottom edge. This is a
deterministic geometry convention, not general reading-order inference. For
example, a full-width title followed by multiple columns may need a future
hierarchical/manual-order representation; this module does not infer one.

By default every source pixel must be selected. A gap, margin-only selection,
or empty selection requires `allow_omissions=True`. Such a plan reports
`uncovered_pixels`; it does **not** assert those pixels are blank. An explicitly
empty plan has zero crops and reports the entire page as omitted. Even this
zero-crop path verifies the source identity before success.

```python
from metricguard.ocr_regions import PixelBox, explicit_region_plan, iter_region_crops

# page is an immutable OcrPageImage returned by load_ocr_pages().
plan = explicit_region_plan(
    page,
    [PixelBox(0, 0, page.width, page.height)],
)
for crop in iter_region_crops(page, plan):
    # crop.source is the full source identity; crop.image is exact cropped RGB.
    assert crop.source == plan.source
```

## Vertical-whitespace heuristic v1

`detect_vertical_regions(page, direction="ltr", white_threshold=255,
min_gutter_width=8, min_region_width=8, limits=None)` is gold-free. It sees only
RGB pixels, never reference text, expected outputs, or recognizer results.

1. A pixel is background only if **all three channels** are at least
   `white_threshold`. A column is background only if every pixel in that column
   is background. The default 255 means exactly white; faint nonwhite pixels
   still count as content.
2. Only internal consecutive background-column runs between occupied columns
   can be gutters. Outer white margins cannot create separate strips.
3. A sufficiently wide gutter proposes a cut at its integer midpoint, rounded
   down. Candidates are considered left-to-right. A cut that would make the
   previous strip or the final remaining strip narrower than `min_region_width`
   is ignored. This changes partitioning, never pixel retention.
4. Full-height strips partition **every source pixel**, including gutters and
   margins. RTL reverses the strip sequence only; it does not mirror pixels or
   change the recognizer's order inside each crop.

The plan outcome is `split`, `no_split`, or `blank`. `blank` means every pixel
met the configured background threshold, not a semantic statement that there
is no text. Both `blank` and `no_split` retain a single full-page crop. A dark
page cannot disappear. Even permissive threshold settings cannot silently drop
faint text. A persisted heuristic plan is recomputed against the supplied
raster before any crop is emitted, so a rehashed but falsely declared detector
outcome is rejected.

The algorithm deliberately cannot detect rotated/perspective pages, overlapping
objects, tables, figures, formulas, paragraphs, hierarchical layouts, or columns
connected by a full-width heading. No confidence score, language inference,
learned detection, deskewing, or general reading-order claim is made. A missed
split falls back to recognizing a larger intact raster, not skipping it.

## Budgets and exact mapping

`OcrRegionLimits(max_regions=64, max_crop_total_pixels=100_000_000)` applies per
plan. Strict hard ceilings are 1,024 regions and 100 million crop pixels; lower
budgets are supported. Existing `OcrPageImage` limits also cap each source page
at 20 million pixels. Count or pixel-budget violations raise an error; there is
no truncation to an apparently successful partial selection.

The overlap check is bounded quadratic in region count; detection is bounded
linear in source pixels. `iter_region_crops` validates the source and detector
once on first iteration, then materializes one region at a time. Cropping copies
exact RGB rows with no interpolation, color conversion, padding, or rotation.
Temporary byte buffers still consume memory: the pixel budgets are **not** a
hard process-RSS or execution-time guarantee. A document consumer must also cap
aggregate pages, regions/provider calls, crop pixels, and output text.

```python
from metricguard.ocr_regions import detect_vertical_regions, iter_region_crops, map_region_result

plan = detect_vertical_regions(page)
for crop in iter_region_crops(page, plan):
    recognition = backend.recognize(crop.image)
    source_lines = map_region_result(crop, recognition)
    # Local (x, y, w, h) becomes (x + left, y + top, w, h).
    # Text and native line/word order are preserved; no region is relabeled as
    # a completed whole-page result. The original recognition remains intact.
```

`map_region_result` rejects results from a different crop even when dimensions
match. It returns source-coordinate `OcrLine` values, not a fabricated
`OcrPageResult` for the full source page. The consumer must retain the plan,
crop-local result, source mapping, provider identity, and explicit omissions.
Translation does not alter the recognizer's angle estimate or deskew its boxes.

## Evidence boundary

`tests/test_ocr_regions.py` uses original synthetic RGB arrays and hand-specified
geometry/pixel oracles. It checks odd gutter midpoint ties, RTL, dark/blank/faint
pages, bridge pixels, ignored narrow cuts, exact row cropping, source-coordinate
translation, nested plan validation, stale identities, omitted coverage, and
budget failures. These tests establish deterministic contracts, **not** OCR
accuracy or equivalence to a full document-layout research pipeline. No model
weights, external dataset, network request, or paid recognizer is used here.
