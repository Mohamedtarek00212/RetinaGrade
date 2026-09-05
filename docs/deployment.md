# RetinaGrade Deployment

The recommended free public deployment is the static ONNX application described
below. The Docker stack remains available for environments that need centralized
server-side inference.

## Optional API container deployment

The container stack exposes the React application and an optional FastAPI API at
`/api`. The React interface still performs ONNX inference in the browser. The API
loads the PyTorch checkpoint once per backend container and is available for
programmatic integrations that require centralized inference.

## Hardware behavior

The browser client uses WebGPU when available and otherwise falls back to CPU.
For requests sent directly to the optional FastAPI service:

- The default CPU target installs CPU-only PyTorch and works on standard hosts.
- The GPU target uses CUDA 12.4 and requires an NVIDIA host with the NVIDIA
  Container Toolkit.
- Both targets use PyTorch 2.5.1; the production requirements intentionally
  exclude training, plotting, evaluation, and TensorBoard packages.
- `RETINAGRADE_DEVICE=auto` selects CUDA when PyTorch can access it and otherwise
  selects CPU.

Keep one Uvicorn worker. Additional workers would each load a separate 175 MB
model copy into RAM or VRAM.

## Model file

Generate the inference checkpoint before starting the stack:

```bash
.venv/bin/python scripts/export_inference_checkpoint.py
```

Compose mounts the result from
`outputs/checkpoints/deployment/model_inference.pt` into the backend as a
read-only file. The checkpoint remains outside the Docker image, which keeps
image builds reproducible and lets deployment platforms attach or download the
model separately.

## CPU stack

```bash
docker compose up --build
```

Open `http://localhost:8080`. Stop the stack with:

```bash
docker compose down
```

## NVIDIA GPU stack

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The GPU image cannot use Apple Silicon's integrated GPU. Test it on the target
Linux/NVIDIA host. If no compatible GPU is exposed to the container, use the CPU
stack instead.

## Single-container Docker image

The `space-cpu` target packages the React frontend, FastAPI application, and
inference checkpoint into one container listening on port `7860`. It remains
available for any host that supports Docker compute.

```bash
docker build --target space-cpu -t retinagrade-space .
docker run --rm -p 7860:7860 retinagrade-space
```

Export the inference checkpoint locally before building this target.

## Free static ONNX deployment

Public demo: https://huggingface.co/spaces/mohamed00212/RetinaGrade

The root README configures Hugging Face as a Static Space. The production React
app lazily downloads `frontend/public/models/retinagrade.int8.onnx`, caches it in
IndexedDB, and runs inference entirely inside the visitor's browser. It tries
WebGPU first and falls back to WebAssembly CPU execution. Uploaded images never
leave the device, and no paid backend is required.

Rebuild the browser model with:

```bash
python -m pip install -e ".[onnx]"
python scripts/export_onnx.py
```

`MatMul` and `Gemm` weights use ONNX Runtime dynamic quantization. Convolution
weights use per-channel INT8 storage followed by `DequantizeLinear`, which keeps
activations in FP32 and avoids the poorly supported `ConvInteger` operator.

The convolution weight pass reduces the browser artifact from 100.0 MiB to
52.8 MiB. A local compatibility benchmark retained the same top-1 prediction on
all 15 curated grade-balanced samples, with a maximum class-probability change
of 0.0169. This small compatibility set does not replace a full validation/test
evaluation when the original image corpus is available.

## Container production settings

- Serve port `8080` behind the hosting platform's HTTPS proxy.
- Store the checkpoint in a private/public model artifact store or persistent
  volume and mount it at `/models/model_inference.pt`.
- Keep uploaded images ephemeral; the API removes each temporary file after the
  request.
- Use platform health checks against `/healthz` for the web container and
  `/health` for the backend container.
