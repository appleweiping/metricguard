# Source-bound multi-region OCR

This is a recognition pipeline, not a metric over supplied transcriptions. It
decodes image frames or PDF pages, constructs an ordered crop plan, copies actual
RGB rectangles, calls the configured `PageOcrBackend` once per planned region,
and translates recognized word geometry to the original rendered page. It never
accepts a gold transcription. The existing `transcribe_document()` v1 whole-page
API and output format are unchanged.

This is not a learned layout model, semantic reading-order system, table parser,
deskewer, rotation correction, or a claim of improved recognition accuracy.
See [region planning](ocr-regions.md) for the deliberately narrow vertical-gutter
heuristic and the explicit, source-bound plan alternative.

## Python API

```python
from metricguard.native_ocr import WindowsOcrBackend
from metricguard.ocr_region_pipeline import (
    OcrRegionPipelineLimits,
    transcribe_region_document,
)
from metricguard.ocr_regions import OcrRegionLimits

# Backend construction performs the configured native capability probe; that
# probe is not a region-recognition call. This example requires Windows OCR.
backend = WindowsOcrBackend(language="en-US")
result = transcribe_region_document(
    "private-input.pdf",
    backend,
    region_limits=OcrRegionLimits(max_regions=16),
    aggregate_limits=OcrRegionPipelineLimits(max_total_regions=200),
    max_document_text_bytes=4 * 1024 * 1024,
)
artifact = result.to_dict()  # Private recognized text and geometry, not redacted.
print(result.digest)
```

Both `transcribe_region_document(path, backend, ...)` and
`transcribe_region_pages(pages, backend, ...)` accept these keyword-only options:

- `planner`: a callable from `OcrPageImage` to `OcrRegionPlan`; `None` selects
  `detect_vertical_regions()` with its default policy. A custom callable can use
  explicit source-bound plans or explicit detector parameters.
- `limits`: `OcrInputLimits` for file/decoded-page limits. The page-stream entry
  point enforces page order, dimensions, total pixels and counts but does not
  possess a source file whose encoded-byte limit it could independently check.
- `region_limits`: `OcrRegionLimits`, applied to each entire page plan.
- `aggregate_limits`: `OcrRegionPipelineLimits` for the complete document.
- `max_document_text_bytes`: structured native-text inventory limit, described
  below; default 16 MiB, maximum 64 MiB.

An explicit planner must return a plan for the exact supplied raster identity.
Crop iteration revalidates the original pixels and declared detector policy.
Missing/extra plan inventories are the responsibility of an inventory wrapper
around a callable; this entry point does not silently consume a separate plan
file. The page-stream caller owns its iterator and must close it if abandoning
it. The file entry point always closes its decoder, including on failure.

## Identities, geometry, and text

The immutable result tree is `OcrRegionDocumentResult.pages` →
`OcrRegionPageResult.regions` → `OcrRegionResult`. A page retains its complete
`plan`, configured `provider`, and `page_id` (plan digest). A region retains its
source-page identity, plan digest, region index/half-open pixel box, exact crop
identity in `recognition.page`, native `recognition`, and content-derived
`region_id`. RGB buffers are not retained in results.

`region.recognition.lines` preserve crop-local boxes. `region.source_lines`
preserve the same word and line sequence and translate each box by the crop's
left/top offset. `region.source_result` is the corresponding source-frame
`OcrPageResult`. There is no scaling or rotation; text angle is preserved, not
used to transform geometry. Source coordinates mean the EXIF-normalized image
or rendered PDF raster, not original JPEG storage orientation or PDF points.

Assembly preserves native line text and order exactly: lines use `"\n"`, regions
use `"\n\n"`, and pages use `"\n\f\n"`. A region's span indexes its page's text;
a page's span indexes document text. All spans are half-open **Unicode code-point**
offsets, not UTF-8 bytes or grapheme clusters. Combining marks, emoji, CRLF,
literal braces and native spacing are not normalized. Separator text belongs to
neither neighboring span. Empty regions/pages have valid zero-width spans;
multiple empty items still contribute their declared separators.

Word boxes are geometry, not invented character offsets. Native word strings
need not equal a whitespace split of the native line text, so no word-text-span
alignment or semantic reading-order guarantee is claimed.

`to_dict()` exports a versioned `metricguard.ocr-region-document.v1` tree.
`OcrRegionDocumentResult.from_dict()` strictly reconstructs closed typed values,
checks crop/source/plan-marker relationships, and recomputes text, spans, status,
counters and identities instead of trusting derived fields. Page and region
types also support `from_dict()`. `result.digest` hashes canonical UTF-8 JSON.

These are consistency checks, not signatures or proof of authentic recognition.
An artifact does not contain original pixels; a self-consistent declared crop
pixel hash cannot independently prove those pixels came from a source image.
For that guarantee, retain the trusted source and rerasterize/replan/recrop,
comparing the resulting source and crop identities. A caller who rewrites both
data and hashes is outside unauthenticated artifact integrity guarantees.

## Empty results are explicit

These distinct situations are never silently collapsed:

- The default detector's `reason="blank"` means all pixels satisfy its declared
  white-threshold policy. It **keeps one full-page crop and calls OCR**. Threshold
  classification is not proof that a page contains no meaningful information.
- `status="recognized_empty"` means the page had planned recognition calls but
  they returned no lines. This can also occur on a nonblank source.
- `status="omitted_all"` means an explicitly empty plan with
  `allow_omissions=True`. It makes zero recognition calls and reports every
  source pixel uncovered. It is not labelled a detected or recognized blank.

An all-empty-recognition or all-explicitly-omitted document is valid if it has at
least one source page. Empty page streams, duplicate/missing/reordered page
indices, or changed source hash/type/renderer are rejected even when no crops
would be recognized. Configured provider identity is retained for zero-call
documents.

Top-level `status="complete"` means every planned crop succeeded, **not** that the
plan selected all source pixels. `coverage` separately reports `complete`,
`partial`, or `omitted_all`, using exact nonoverlapping rectangle areas.

## Admission, counters, and failures

Each complete page plan is admitted before its first recognition call. Both
per-plan region/pixel budgets and cumulative document region/pixel budgets are
checked then; there is no partially admitted page. Later raster-stream failures
can occur after earlier calls. File decoding additionally preflights all page
sizes through the existing loader.

Aggregate defaults are 1,000 planned regions, 100 million crop pixels, 100,000
words and 100,000 lines. Hard ceilings are 10,000 regions, 200 million crop pixels,
and 100,000 words/lines. Input limits independently cap decoded source pixels.
Boolean/noninteger/out-of-range budgets are rejected before source iteration.
Known separator bytes are admitted before recognition; unknown result text and
word/line counts are checked immediately after each call, before continuing.

`structured_text_bytes` counts every native line string, every native word string,
native line separators, and the region/page assembly separators once. It is not
just the displayed document string. The serialized artifact additionally contains
duplicated native/source-frame geometry and text, repeated metadata and JSON
framing; the **entire** exported contract has a separate 64 MiB bound. Public
constructors admit aggregate word/line counts and a native-text byte lower bound
before joining page/document text or expanding nested geometry dictionaries.
Constructors and deserializers reject an over-limit envelope rather than truncating it. A
positive OCR result may fit its native page limit but fail this larger enclosing
document admission check.

Counters report source pages/pixels, planned/completed regions, crop pixels,
uncovered pixels, lines/words, structured text, and the three empty-page categories.
`recognition_calls` counts calls issued by this sequential pipeline, not backend
capability probes, internal retries or external service billing units. The
pipeline itself performs no retries.

Provider identity is checked before/after planning, before/after every recognition
and at page/document completion. Each result must bind the exact actual crop.
Malformed, wrong-source, wrong-provider, changed-identity or failed output aborts
the operation. No partial document is returned or written. Earlier provider
effects cannot be rolled back, and a caller's explicit retry can repeat them.
The API creates no files and has no hidden network/model/download behavior;
effects of a caller-configured backend remain that caller's responsibility.

## Cost and verification limits

For decoded pixels `P`, copied crop pixels `C`, regions `R`, and structured output
size `S`, work includes raster hashing/scanning `O(P)`, crop copies `O(C)`, existing
plan overlap/order checks and repeated plan hashing up to `O(R²)`, serialization
and geometry processing `O(S)`, plus backend recognition cost. Detector plans are
checked against pixels again before cropping. The pipeline caches source and
plan identities per page; it does not rehash the full source raster for each
recognized region.

Processing retains one source page and one active crop plus the complete bounded
result inventory. Encoded input snapshots, decoder/native buffers, Python object
overhead and transient serialization copies are additional. These are admission
limits, not hard RSS, wall-clock or recognition-quality guarantees.

Portable tests use authored RGB/PNG/raster-only PDF inputs and scripted recognition
outputs, with independent arithmetic for crops, translated boxes, Unicode spans,
counts, boundary admission and digest calculations. They are engineering tests,
not gold OCR accuracy evaluation or evidence of parity with a research OCR system.
