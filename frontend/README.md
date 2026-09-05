# RetinaGrade Frontend

React and TypeScript research interface that runs the RetinaGrade ONNX model
entirely inside the browser. The visual tokens follow `docs/design-system.md`.

The result workspace presents class probabilities, referable-DR likelihood,
probability-weighted expected grade, confidence margin, ordinal thresholds,
image-quality feedback, measured inference time, and a class activation map.
The explanation view overlays the areas that contributed most strongly to the
selected class and pairs them with confidence-aware research guidance. It does
not identify lesions, establish a diagnosis, or determine clinical urgency.

The reporting workspace generates two A4 PDFs. The clinician report includes
the detailed prediction, probabilities, ordinal thresholds, image quality,
focus map, notes, and sign-off fields. The patient summary uses simpler wording
and requires the reviewing clinician to confirm review and enter the intended
next steps. Report generation is lazy-loaded and remains entirely on-device.

## Local development

Start the frontend:

```bash
cd frontend
npm ci
npm run dev
```

Vite uses `http://127.0.0.1:5173` by default and automatically selects the next
available port during local development. No API server is required.

The approximately 53 MB quantized model is stored at
`public/models/retinagrade.int8.onnx`. It is downloaded only when the visitor
selects a valid image, then cached in IndexedDB. Download and runtime preparation
begin while the user reviews the image, and the cache write does not block
session creation. ONNX Runtime Web selects WebGPU when available and falls back
to WebAssembly on CPU. Image preprocessing, quality checks, inference, and
result calculations all stay on the device.
The explanation map is generated in the same model pass, so it does not require
another upload or a separate server request.

## Production build

```bash
npm run build
```

The static production output is written to `frontend/dist/`.

Append `?cpu=1` to the URL to force the WebAssembly path during compatibility
testing.
