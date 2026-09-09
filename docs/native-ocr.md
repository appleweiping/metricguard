# Local image and PDF transcription

This **unreleased development-branch** feature executes real local OCR. It is separate from
MetricGuard's older `ocr`, `ocr-backend` and `ocr-document` evaluation commands, which score
provided predictions or caller-supplied transcribers. `transcribe` requires no reference text.

## Install and run

Install the same development source as the README, with the imaging extra:

```bash
python -m pip install "metricguard[ocr] @ git+https://github.com/appleweiping/metricguard.git@feat/whole-repository-alignment"
metricguard transcribe scan.png --language en-US --output scan.ocr.json
metricguard transcribe scanned-document.pdf --language en-US --dpi 200 --max-pages 20 --output document.ocr.json
```

The included recognizer requires Windows, Windows PowerShell 5.1, and an **already-installed**
Windows OCR language. `WindowsOcrBackend().available_languages` reports installed languages after
initializing the selected language. Unsupported platforms and missing languages fail explicitly.
Nothing installs a language, fetches model weights, or calls a network service. Pillow and
pypdfium2 are optional Python dependencies, not OCR models. The metric APIs remain usable without
either extra on Python 3.10–3.14 across platforms.

```python
from metricguard import OcrInputLimits, WindowsOcrBackend, transcribe_document

backend = WindowsOcrBackend("en-US", timeout=30)
document = transcribe_document("scan.pdf", backend, limits=OcrInputLimits(dpi=200, max_pages=20))
print(document.text)
print(document.digest)
print(document.pages[0].lines[0].words[0].box)  # Guard empty pages in applications.
```

## What happens

1. Snapshot one input file and hash its exact bytes. Detect PDF by signature; otherwise decode
   PNG, JPEG, TIFF, BMP or WEBP, including supported multiframe images.
2. Preflight page count, dimensions and aggregate raster pixels before any recognition. Apply
   image EXIF orientation and compose transparency onto white. PDF pages are rendered, never
   read through their text layer. Produce canonical RGB bytes and a dimension-aware pixel hash.
3. Give an explicitly chosen `PageOcrBackend` only `OcrPageImage`, never a gold transcription.
   The native backend sends a PNG through bounded stdin to a fixed packaged helper. The helper
   invokes `Windows.Media.Ocr.OcrEngine.RecognizeAsync`, not a shell command supplied by input.
4. Validate returned language, provider identity, finite word geometry and bounds. Keep native
   line and word ordering, derive each line's enclosing box, and release raster buffers.
5. Assemble all pages in input order, including blank pages. Publish JSON only after all succeed.

`OcrPageImage`, identities, words, lines and results are frozen values. List inputs are copied
into immutable tuples. Page indices are zero-based. Boxes are `[x, y, width, height]` in the
**decoded, EXIF-oriented or PDF-rasterized image's pixel coordinates**, origin at the upper left;
they are not coordinates in PDF points or in the original pre-orientation image.
`text_angle` is the engine's detected angle in degrees or null. No confidence score is invented.
Line boxes enclose word boxes; no paragraph, table, formula or document-layout hierarchy is inferred.

`OcrDocumentResult.text` joins native line text with newlines and page text with `"\n\f\n"`.
`page_spans` are exact Unicode-code-point `[start,end)` intervals excluding separators. Blank
pages have zero-width spans, so a successful blank page is distinguishable from a missing page.
The JSON format is `metricguard.ocr-document.v1`; its digest hashes canonical JSON including
identities, geometry and text. No raw raster or local source path is included in the result.

## Identity and failure semantics

Source hashes identify exact input bytes; pixel hashes identify RGB pixels plus width/height.
Renderer identity includes Pillow or pypdfium2 version and processing settings. Provider identity
pins the packaged helper SHA-256, explicit language, timeout/transport limits, OS version and
reported language/dimension capabilities. It is checked before and after each recognition.
**Windows does not expose model-weight hashes**: OS/capability identity is not byte-identical
model reproducibility. Use stored outputs and hashes to audit what actually ran.

`transcribe_document` returns a complete document or raises; it never returns a misleading
partial success. Earlier recognizer calls cannot be rolled back, and a custom local provider can
have its own effects. The CLI writes through a same-directory temporary file and atomic replace;
decode, recognition and publication failures preserve an existing destination. Same-path,
resolved-symlink and hardlink aliases of the input are rejected before engine initialization.
This assumes caller-controlled paths/directories, not an adversary racing filesystem operations.

The native subprocess uses a fixed interpreter/helper argv with `shell=False`, a hidden window,
bounded requests, concurrent bounded stdout/stderr readers, finite execution timeout and cleanup.
Output overflow kills the helper. Errors do not include raw native stderr, document contents or
helper exception messages. The helper does not launch descendants; this is not a supervisor for
arbitrary user commands. The older `CommandOcrBackend` is a separate caller-command adapter.

## Resource and scope limits

Defaults: 64 MiB source file, 100 pages, 10,000 pixels per dimension, 20 million pixels per page,
100 million pixels per input, 200 DPI for PDFs, 30 seconds per native helper invocation, 4 MiB per
native output pipe, 4 MiB structured text per page, 16 MiB structured text per document, and
100,000 recognized words per document. Text budgets count both line and word UTF-8 strings plus
assembly separators, so empty line strings cannot hide oversized word payloads. The
CLI's complete JSON report is capped at 64 MiB. Python options can lower limits; declared upper
ceilings prevent unbounded configuration. Input and native transport are bounded during reading,
not only after collecting a full response. Raster data is processed sequentially, one page at a time.

These are **file, pixel, text and transport bounds, not a hard total-memory sandbox**. Decoder
libraries may allocate native memory while parsing compressed inputs; pixel limits do not bound
PDFium parsing cost, Pillow metadata, or WinRT process RSS. PDF rasterization has no hard wall-clock
timeout. Only helper recognition/probing has a subprocess timeout. Use trusted documents or a
separate OS-isolated worker with its own CPU/memory limits for hostile inputs. PDFium is not
thread-safe: MetricGuard serializes its PDFium calls, but cannot coordinate unrelated PDFium calls
made by other libraries in the same process. Engine reading order may be unsuitable for complex
layouts. There is no custom trained model, GPU execution, service deployment, resumable OCR cache,
automatic language detection, region detector, or handwriting/multilingual accuracy guarantee.

## Reproduce the narrow verification

From a development checkout with `.[dev]` installed and en-US OCR already available:

```powershell
$env:METRICGUARD_RUN_NATIVE_OCR = "1"
python -m pytest tests/test_native_ocr.py -m native_ocr --no-cov
python benchmarks/verify_native_ocr.py --fixtures authored-ocr-demo --output native-ocr.json
metricguard transcribe authored-ocr-demo/two-pages.pdf --output authored-ocr-demo/result.json
```

The fixture directory must not already contain the three fixture names. Omitting `--fixtures`
uses automatically cleaned temporary files. The script draws two original, clean English text
rasters with Pillow's default font, embeds those rasters in a two-page PDF with **no text layer**,
and adds a blank PNG. It invokes the real engine before scoring CER/WER. The checked-in
[verification result](../benchmarks/results/native-ocr.json) records fixture/runtime hashes,
provider capabilities, measurements and timings. It is a synthetic integration check on two
distinct texts, not an independently collected OCR dataset or broad accuracy claim. `tracemalloc`
excludes fixture creation and does not measure native allocations or child-process RSS.

Portable tests use real image/PDF decoders and fake provider transport to exercise failure
contracts; they do not claim real recognition. The native integration test is skipped unless
explicitly enabled on Windows. Windows language availability is not assumed in CI.

## Primary API references

- [Microsoft: OcrEngine.RecognizeAsync](https://learn.microsoft.com/en-us/uwp/api/windows.media.ocr.ocrengine.recognizeasync?view=winrt-26100)
- [pypdfium2: Python API, resource management and thread safety](https://pypdfium2.readthedocs.io/en/stable/python_api.html)
- [Pillow: image limits and decompression warnings](https://pillow.readthedocs.io/en/stable/reference/Image.html)
