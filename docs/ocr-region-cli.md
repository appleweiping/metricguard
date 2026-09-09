# Local region OCR command

`transcribe-regions` decodes an image or rasterizes every PDF page, selects ordered
pixel regions, executes the installed Windows recognizer on each crop, and assembles
one source-bound document. This is transcription without reference text, not an
OCR accuracy evaluation. It does not download models, install language packs,
contact a provider, or execute a caller-supplied command/plugin.

These are unreleased development-branch features. Install the same branch as the
examples, with the optional image/PDF dependencies:

```powershell
python -m pip install "metricguard[ocr] @ git+https://github.com/appleweiping/metricguard.git@feat/whole-repository-alignment"
metricguard transcribe-regions document.pdf --language en-US --output private-report.json
```

The backend requires Windows, PowerShell 5.1, and an already installed recognizer
language. Other platforms fail with a controlled exit code; the pure planning and
result APIs remain portable. The existing `transcribe` command is unchanged.

## Detection and order

The default `vertical-whitespace.v1` detector tests RGB pixels, not recognized text.
It partitions the entire raster at qualifying full-height background-column gaps.
Defaults are LTR order, white threshold 255, minimum gutter width 8 pixels, and
minimum region width 8 pixels. All original pixels remain in the resulting strips.
Blank pages and pages with no valid split retain a single full-page region; blank
detector output still goes through the real recognizer.

```powershell
metricguard transcribe-regions columns.png --direction rtl --white-threshold 255 --min-gutter-width 100 --min-region-width 100 --output private-columns.json
```

LTR means left-edge-first, then top; RTL means right-edge-first, then top. Direction
changes strip order, not the native recognizer's word order inside a crop. This is
a narrow whitespace heuristic: inter-word gaps can also become cuts. It does not
infer semantic reading order, detect tables, deskew pages, recognize handwriting
as a distinct model, or promise better accuracy than whole-page OCR. Tune pixel
thresholds for the actual raster resolution; PDF DPI changes that resolution.

## Source-bound plan files

Instead of detector options, `--plans private-plans.json` accepts this exact envelope:

```json
{"format":"metricguard.ocr-region-plans.v1","plans":[]}
```

The empty array above shows the envelope shape only: an executable file must contain
one complete `OcrRegionPlan.to_dict()` record per source page. Pages must be listed
exactly once, with indices 0 through N−1, in source order. No extra fields, duplicate
JSON keys, nonfinite numbers, or malformed UTF-8 are accepted. Detector flags cannot
be combined with `--plans`; they are never silently ignored or used to override it.

Generate complete plans from the same source and ingestion settings, for example:

```python
import json
from contextlib import closing
from pathlib import Path

from metricguard.ocr_inputs import OcrInputLimits, load_ocr_pages
from metricguard.ocr_regions import detect_vertical_regions

with closing(load_ocr_pages("columns.png", limits=OcrInputLimits(dpi=200))) as pages:
    plans = [detect_vertical_regions(page, min_gutter_width=100).to_dict() for page in pages]
with Path("private-plans.json").open("x", encoding="utf-8") as stream:
    json.dump({"format": "metricguard.ocr-region-plans.v1", "plans": plans}, stream)
```

Each record binds file SHA-256, page index/type, raster dimensions/pixel SHA-256,
renderer identity, exact half-open pixel boxes, ordering, detector settings or
explicit selection, coverage counts and plan digest. Plans are reparsed and checked
against the actual rasterized page before that page's crops are recognized. A
serialized detector plan must also reproduce the declared detector result.

For explicit boxes use `explicit_region_plan(page, boxes, ...)` with `PixelBox`
values. Omissions require `allow_omissions=True`; they remain visible in coverage
and uncovered-pixel counts. An empty explicit selection is `omitted_all`, not a
successfully recognized blank page. There is no implicit filling, reordering or
discarding of unsupported boxes.

Invalid declared plan budgets are rejected before backend construction. A missing,
extra or mismatched later page rejects the entire report, but earlier local
recognizer calls may already have happened. There is no transactional rollback of
recognition, cache/resume protocol, or promise of zero work before every mismatch.

## Limits

All limits are explicit CLI options. They reject work rather than truncating output.

| Scope | Options and defaults |
| --- | --- |
| Input bytes/pages | `--max-file-bytes 67108864`, `--max-pages 100` |
| Raster dimensions | `--dpi 200`, `--max-dimension 10000`, `--max-page-pixels 20000000` |
| Input aggregate | `--max-total-pixels 100000000` |
| Per-page region plan | `--max-regions 64`, `--max-crop-total-pixels 100000000` |
| All page crops | `--max-total-regions 1000`, `--max-total-crop-pixels 100000000` |
| Recognized structures | `--max-document-words 100000`, `--max-document-lines 100000`, `--max-document-text-bytes 16777216` |
| Each native exchange | `--timeout 30`, `--max-native-output-bytes 4194304` |

Hard ceilings are 1000 pages, 20 million pixels per page, 200 million input pixels,
1024 regions per page, 10,000 regions/200 million crop pixels per document,
100,000 words/lines, and 64 MiB of structured text. Text limits include both line
and word strings plus assembly separators, so hidden word text cannot bypass them.
Native timeout is in (0, 300] seconds and native output is at most 16 MiB per
stdout/stderr pipe. It is a per-exchange timeout, not a whole-document deadline.

The plan file is capped at 16 MiB and each typed plan at 1 MiB; the final serialized
CLI envelope is capped at 64 MiB. These are encoded-data/decoded-raster admission
checks, not a hard process RSS or CPU sandbox. JSON encoding may allocate a complete
existing string chunk; Pillow, PDFium and Windows OCR have their own native memory
allocations. Do not treat this CLI as safe for arbitrary hostile files on an
untrusted public server.

## Private output and publication

On success, stdout or a new `--output` file receives one UTF-8 JSON envelope:

```json
{"format":"metricguard.ocr-region-transcription.v1","private_data":true,
 "document":{},"document_digest":"..."}
```

This is a schematic, not a valid document fixture. The nested document is the full
strict `OcrRegionDocumentResult.to_dict()` with `status`, coverage, source/provider
identities, plans, crop recognition, source-coordinate line/word boxes, exact text
and page/region character spans. Its digest is independently recomputable. Character
spans index Python Unicode code points, not UTF-8 bytes; word boxes are geometric
and do not claim a word-to-character alignment.

`status: complete` means every selected crop completed. Inspect `coverage` separately:
it can be `complete`, `partial` or `omitted_all`. It is not an accuracy score. Reports
contain private recognized content; a privacy flag does not redact that content.
Runtime errors are summarized without printing raw exception text or source content.

An output path must be new, must not alias the input or plan through resolved paths,
symlinks or hard links, and must have an existing parent directory. No directory is
created implicitly. The complete bounded report is flushed/fsynced to a private
temporary file, then installed with an exclusive hard link. Existing or racing
destinations are not overwritten. A filesystem without hard-link support fails
instead of falling back to overwrite. Use a trusted local directory: these checks
do not defeat an adversary concurrently replacing directory ownership or topology.

No partial report is published after a recognition or validation failure. If hard-link
installation succeeds but temporary-file cleanup fails, exit 2 warns that an artifact
or private temporary alias may exist. Inspect and clean up that private alias before
retrying; successful publication is not rolled back.

Exit 0 means recognition/publication completed; exit 2 means usage, configuration,
recognition, validation or I/O failure. A closed stdout consumer after successful
recognition retains exit 0 and attempts a diagnostic on stderr; the command cannot
guarantee the consumer received all bytes. With `--output`, successful publication
writes no duplicate report to stdout.

The opt-in native CLI integration test uses authored two-column PNG and raster-only
two-page PDF fixtures. Its text oracle ignores recognizer-added whitespace while
checking character content and region/page order; it does not establish zero CER/WER
or quality on a real-world corpus. Literal recognized strings are retained unchanged.
