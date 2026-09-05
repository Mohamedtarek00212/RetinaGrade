import * as ort from "onnxruntime-web/webgpu";

import { preprocessImage } from "./preprocess";
import type { BrowserPrediction, ModelProgress } from "./types";

const MODEL_VERSION = "retinagrade-int8-v1";
const MODEL_URL = `${import.meta.env.BASE_URL}models/retinagrade.int8.onnx`;
const MODEL_DB = "retinagrade-model-cache";
const STORE = "models";
const LABELS = ["No DR (Normal)", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR (PDR)"];

type RuntimeSession = { session: ort.InferenceSession; runtime: "WebGPU" | "CPU" };

let sessionPromise: Promise<RuntimeSession> | null = null;

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(MODEL_DB, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function readCachedModel(): Promise<ArrayBuffer | null> {
  try {
    const database = await openDatabase();
    return await new Promise((resolve, reject) => {
      const request = database.transaction(STORE).objectStore(STORE).get(MODEL_VERSION);
      request.onsuccess = () => resolve((request.result as ArrayBuffer | undefined) ?? null);
      request.onerror = () => reject(request.error);
    });
  } catch {
    return null;
  }
}

async function cacheModel(buffer: ArrayBuffer) {
  try {
    const database = await openDatabase();
    await new Promise<void>((resolve, reject) => {
      const request = database.transaction(STORE, "readwrite").objectStore(STORE).put(buffer, MODEL_VERSION);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  } catch {
    // Storage quotas vary by browser; inference still works without persistence.
  }
}

async function downloadModel(onProgress: (progress: ModelProgress) => void) {
  const cached = await readCachedModel();
  if (cached) {
    onProgress({ label: "Loading cached model" });
    return cached;
  }

  const response = await fetch(MODEL_URL);
  if (!response.ok || !response.body) throw new Error("Could not download the on-device model.");
  const total = Number(response.headers.get("content-length") ?? 0);
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    onProgress({
      label: "Downloading model",
      percent: total ? Math.min(100, Math.round((received / total) * 100)) : undefined,
    });
  }

  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  await cacheModel(bytes.buffer);
  return bytes.buffer;
}

async function createSession(onProgress: (progress: ModelProgress) => void): Promise<RuntimeSession> {
  const model = await downloadModel(onProgress);
  onProgress({ label: "Preparing inference engine" });

  ort.env.wasm.numThreads = crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  const forceCpu = new URLSearchParams(window.location.search).has("cpu");
  if ("gpu" in navigator && !forceCpu) {
    try {
      const session = await ort.InferenceSession.create(new Uint8Array(model), {
        executionProviders: ["webgpu"],
        graphOptimizationLevel: "all",
      });
      return { session, runtime: "WebGPU" };
    } catch {
      onProgress({ label: "Using CPU compatibility mode" });
    }
  }

  const session = await ort.InferenceSession.create(new Uint8Array(model), {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  return { session, runtime: "CPU" };
}

function softmax(values: readonly number[]) {
  const maximum = Math.max(...values);
  const exponentials = values.map((value) => Math.exp(value - maximum));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
}

function sigmoid(value: number) {
  return 1 / (1 + Math.exp(-value));
}

export async function predictOnDevice(
  file: File,
  onProgress: (progress: ModelProgress) => void,
): Promise<BrowserPrediction> {
  onProgress({ label: "Processing image" });
  const { tensor, quality } = await preprocessImage(file);
  if (!sessionPromise) {
    sessionPromise = createSession(onProgress).catch((error) => {
      sessionPromise = null;
      throw error;
    });
  }
  const { session, runtime } = await sessionPromise;
  onProgress({ label: `Analyzing on ${runtime}` });

  const startedAt = performance.now();
  const outputs = await session.run({ image: new ort.Tensor("float32", tensor, [1, 3, 512, 512]) });
  const inferenceMs = performance.now() - startedAt;
  const classificationLogits = Array.from(outputs.classification_logits.data as Float32Array);
  const ordinalLogits = Array.from(outputs.ordinal_logits.data as Float32Array);
  const probabilities = softmax(classificationLogits);
  const ordinalProbabilities = ordinalLogits.map(sigmoid);
  const grade = probabilities.indexOf(Math.max(...probabilities));
  const ranked = [...probabilities].sort((left, right) => right - left);
  const confidenceMargin = ranked[0] - ranked[1];

  return {
    grade,
    label: LABELS[grade],
    confidence: probabilities[grade],
    probabilities,
    ordinal_probabilities: ordinalProbabilities,
    assessment_status:
      probabilities[grade] >= 0.5 && confidenceMargin >= 0.15
        ? "conclusive"
        : "review_recommended",
    confidence_margin: confidenceMargin,
    inference_ms: inferenceMs,
    quality,
    runtime,
  };
}
