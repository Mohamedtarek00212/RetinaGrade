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

### Automatic deployment from GitHub

`.github/workflows/tests.yml` is the production release path. Pull requests run
the Python and frontend checks. A push to `main` publishes only after both test
jobs pass:

1. Git LFS downloads the browser ONNX model.
2. `npm ci`, ESLint, and the production build run in GitHub Actions.
3. The tested `frontend/dist` artifact is uploaded to `dist` in the Hugging Face
   Space.
4. The workflow waits for the public static URL to expose the new hashed bundle.

Create a fine-grained Hugging Face token with write access to
`mohamed00212/RetinaGrade`, then add it to the GitHub repository under
**Settings > Secrets and variables > Actions** as a repository secret named
`HF_TOKEN`. Never commit the token to this repository or add it as a frontend
environment variable.

The deployment job is attached to the GitHub `production` environment, whose
URL points to the direct static Space address. The workflow uploads only the
generated static site; training checkpoints, datasets, source reports, and
other repository content are not transmitted to the Space.

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

## Browser explanation output

The ONNX graph also returns a class activation map for the predicted grade. The
model uses global average pooling followed by a linear neck and classification
head, so export combines the two linear weight matrices and applies the selected
class weights to the final PLKA spatial feature map. ReLU and per-image min-max
normalization produce the displayed contribution overlay in the same inference
pass.

The map is an explanation of the model's spatial contribution, not lesion
localization or a diagnosis. The interface therefore presents it alongside the
top-two probability margin, image-quality findings, and conservative guidance
to seek qualified eye-care review when the output is uncertain or non-zero.

The explanation-enabled model retained identical classification logits and
top-1 predictions against the previous browser model on all 15 curated samples.
Every tested activation map had a normalized range from 0 to 1 and non-zero
spatial variation.

## On-device PDF reports

The result workspace can generate a detailed clinician review report and a
plain-language patient information summary. The patient document is gated by a
clinician-review confirmation and clinician-authored next steps. Both documents
include explicit research-use limitations, and neither makes an automated care
decision. Names, identifiers, notes, images, and generated PDFs stay inside the
browser and are not sent to the static host.

The PDF library is split into lazy production chunks, so it is downloaded only
when a report is requested and does not add to the initial application load.

## Container production settings

- Serve port `8080` behind the hosting platform's HTTPS proxy.
- Store the checkpoint in a private/public model artifact store or persistent
  volume and mount it at `/models/model_inference.pt`.
- Keep uploaded images ephemeral; the API removes each temporary file after the
  request.
- Use platform health checks against `/healthz` for the web container and
  `/health` for the backend container.
