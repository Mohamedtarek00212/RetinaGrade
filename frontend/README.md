# RetinaGrade Frontend

React and TypeScript research interface that runs the RetinaGrade ONNX model
entirely inside the browser. The visual tokens follow `docs/design-system.md`.

The result workspace presents class probabilities, referable-DR likelihood,
probability-weighted expected grade, confidence margin, ordinal thresholds,
image-quality feedback, and measured inference time. These are model-derived
research outputs and are explicitly not presented as clinical decisions.

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
starts an analysis, then cached in IndexedDB. ONNX Runtime Web selects WebGPU
when available and falls back to WebAssembly on CPU. Image preprocessing,
quality checks, inference, and result calculations all stay on the device.

## Production build

```bash
npm run build
```

The static production output is written to `frontend/dist/`.

Append `?cpu=1` to the URL to force the WebAssembly path during compatibility
testing.
