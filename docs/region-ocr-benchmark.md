# Region OCR: authored-layout engineering benchmark

This benchmark runs the installed Windows recognizer on original synthetic
images, not a replay provider. It compares ordinary full-page recognition with
the [region-first pipeline](ocr-region-pipeline.md), retaining the original text
from both. It does not measure a real-world document distribution or establish
equivalence to a complete OCR research system.

## Reproduce from the development branch

Use `feat/whole-repository-alignment` and install its optional imaging dependencies.
The Windows English OCR language must already be installed; this script neither
installs a language/model nor chooses a paid or remote API. From the checkout:

```shell
python -m pip install -e ".[ocr]"
python benchmarks/verify_region_ocr.py --output ./new-region-ocr.json
```

The output must be a **new file** in an existing directory. By default, original
fixtures are created in an owned temporary directory and removed after the run.
To retain them, specify `--fixtures ./new-authored-fixtures`; its parent must exist
and that directory must be new. Failures can leave a partial fixture directory.
The JSON report is opened exclusively, not atomically renamed: a disk/write failure
can leave a partial new report. Inspect it before retrying into another new path.
This differs from the [production CLI's publication contract](ocr-region-cli.md).

## Fixed test inventory

Five source-page evaluations come from four files, not five independent samples:

- `columns.png`: two columns, each with two original uppercase English lines.
- `columns-and-blank.pdf`: the same column image followed by an all-white page.
  Both pages are checked to have no embedded text layer before recognition.
- `bridged-columns.png`: the column image with a full-width black rule. It
  deliberately defeats a detector requiring an uninterrupted full-height gutter.
- `single.png`: a separate two-line, single-column image, exercising no-split fallback.

The generator uses Pillow's bundled `ImageFont.load_default(size=42)` at fixed
positions on 1200 by 320 RGB canvases. The report records dependency versions,
fixture hashes, rasterization policy through source identities, native provider
identity, and detector settings: white threshold 255, minimum gutter 100 pixels,
minimum region width 200 pixels. This configuration is fixed before execution,
not searched to maximize measured recognition accuracy. PDF rasterization uses
200 DPI; the input PDF is saved with a 144-DPI raster scale.

The expected inventory is five full-page calls and seven region calls. All-white
input still reaches the recognizer. Backend initialization/capability probes are
not counted as recognition calls. A detector's `blank` label is not evidence that
the recognizer returned empty output: hallucinated text is retained, as are wrong
words, spaces and reading order. The bridged negative case must remain no-split;
the benchmark does not silently switch to manual boxes to improve its score.

## Independent checks and scoring

References are consulted only after both recognitions. The detector and backend
receive source identities and RGB, not a reference transcript. The script
independently slices RGB row bytes from a fresh decoding of each original source,
recomputes each crop's pixel identity, translates every native word box by its
integer crop origin, and checks region/page Unicode-codepoint text spans. Source
page identities must agree between baseline and region execution. These checks
do not assert that native word geometry is linguistically correct.

Each page retains raw and normalized exact-match flags, character error rate,
word error rate and their explicit edit-distance/reference-length denominators.
Both rates use the reported `TextNormalizer()` policy: NFC, collapsed/trimmed
whitespace, preserved case and punctuation. This differs from the library's
default WER convenience policy, which lowercases and strips punctuation. Raw text
and raw exact match remain present; normalization never changes stored recognition.
If blank gold has nonempty prediction, CER/WER are explicitly undefined with a
reason, not zero. All scores are retained; improvement is not an assertion.

Every package Python source and native helper, the benchmark script and the
fixture files are fingerprinted before and after execution. Changed files fail
the measurement. Run in a frozen checkout and fresh process; file hashes alone
do not authenticate a machine or prove arbitrary previously loaded code identity.

Per-fixture timings are single observed decode/recognition runs, not a statistical
throughput benchmark. `tracemalloc` includes Python allocations during recognition
and independent checks, excluding fixture generation and backend initialization;
it does not include Pillow/PDFium native allocations, PowerShell child RSS or the
Windows engine. No claim about broad accuracy, learned layout detection, tables,
historical manuscripts, languages beyond the fixture, or deployment scale follows.

Portable tests also replace the recognizer with deliberately wrong typed results,
including text hallucinated on the blank page. Those tests verify that unfavorable
outcomes survive reporting; they are not the actual native measurement.

## Recorded local run

The [retained result](../benchmarks/results/region-ocr.json) contains five page
evaluations from the inventory above. Both full-page and region recognition had
normalized CER/WER of zero on these clean fixtures: **no normalized recognition
accuracy improvement was observed**. Region output adds the declared blank-line
separator between columns; this accounts for the raw exact-match difference,
not better recognition of the words.

The full-width-rule case correctly remains unsplit under this limited detector,
although the native recognizer already recovers the fixture's reading order.
The blank page produced no text in this particular run. Independent checks
reconstructed all seven crops, translated 28 word boxes, and checked all five
page spans and seven region spans. Five full-page plus seven crop calls occurred;
the measured Python-only peak was 10,714,191 bytes. These are engineering results
for this one authored run, not a performance or real-world accuracy claim.
