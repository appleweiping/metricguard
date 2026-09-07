# Document-level OCR

MetricGuard can evaluate an OCR backend over ordered pages while retaining both
global page metrics and one report per document. This is useful when a document
is split into pages by a separate detector or manifest builder: the backend is
still a dependency-injected callable, so MetricGuard does not claim to provide
an image model or a cloud OCR service.

The manifest accepts JSON or JSONL records with `id`, `document_id`, `page`,
`reference`, and an optional `image` path. Page paths are resolved relative to
the manifest. Duplicate page IDs and duplicate document/page positions are
rejected before any backend call.

```json
{"id":"invoice-1-p0","document_id":"invoice-1","page":0,
 "image":"pages/invoice-1.png","reference":"total 12.00"}
```

Run the shell-free command adapter once per argv token:

```bash
metricguard ocr-document pages.jsonl \
  --command tesseract --command {image} --command stdout \
  --output document-report.json
```

The JSON report contains the page-level CER/WER/exact-match report and sorted
`document_reports`. The Python API is `run_document_ocr_benchmark()` and accepts
a callback receiving an `OcrDocumentPage`, which allows a model client or a
deterministic fixture backend to be injected without changing metric semantics.
