import type { ImageQuality } from "./types";

const INPUT_SIZE = 512;
const DETECTION_MAX_SIDE = 1024;
const MEAN = [0.3858112836164846, 0.2040913117643135, 0.06115751019223669];
const STD = [0.2949677943634556, 0.16129463634063174, 0.07802564237306733];

type Bounds = { x: number; y: number; width: number; height: number };

function canvas(width: number, height: number) {
  const element = document.createElement("canvas");
  element.width = width;
  element.height = height;
  return element;
}

function largestComponent(mask: Uint8Array, width: number, height: number): Bounds | null {
  const queue = new Int32Array(width * height);
  let bestArea = 0;
  let best: Bounds | null = null;

  for (let start = 0; start < mask.length; start += 1) {
    if (mask[start] === 0) continue;
    let head = 0;
    let tail = 1;
    queue[0] = start;
    mask[start] = 0;
    let area = 0;
    let minX = width;
    let minY = height;
    let maxX = 0;
    let maxY = 0;

    while (head < tail) {
      const index = queue[head++];
      const y = Math.floor(index / width);
      const x = index - y * width;
      area += 1;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);

      for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
        const nextY = y + offsetY;
        if (nextY < 0 || nextY >= height) continue;
        for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
          if (offsetX === 0 && offsetY === 0) continue;
          const nextX = x + offsetX;
          if (nextX < 0 || nextX >= width) continue;
          const next = nextY * width + nextX;
          if (mask[next] === 0) continue;
          mask[next] = 0;
          queue[tail++] = next;
        }
      }
    }

    if (area > bestArea) {
      bestArea = area;
      best = { x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1 };
    }
  }

  return best;
}

function analyzePixels(data: Uint8ClampedArray, width: number, height: number) {
  const pixels = width * height;
  const visible = new Uint8Array(pixels);
  const gray = new Float32Array(pixels);
  let visibleCount = 0;
  let brightnessSum = 0;
  let redSum = 0;
  let greenSum = 0;
  let blueSum = 0;

  for (let index = 0; index < pixels; index += 1) {
    const offset = index * 4;
    const red = data[offset];
    const green = data[offset + 1];
    const blue = data[offset + 2];
    const value = 0.299 * red + 0.587 * green + 0.114 * blue;
    gray[index] = value;
    if (value > 10) {
      visible[index] = 1;
      visibleCount += 1;
      brightnessSum += value;
      redSum += red;
      greenSum += green;
      blueSum += blue;
    }
  }

  let laplacianSum = 0;
  let laplacianSquareSum = 0;
  let laplacianCount = 0;
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const index = y * width + x;
      const value =
        gray[index - width] + gray[index + width] + gray[index - 1] + gray[index + 1]
        - 4 * gray[index];
      laplacianSum += value;
      laplacianSquareSum += value * value;
      laplacianCount += 1;
    }
  }
  const laplacianMean = laplacianCount ? laplacianSum / laplacianCount : 0;
  const sharpness = laplacianCount
    ? laplacianSquareSum / laplacianCount - laplacianMean * laplacianMean
    : 0;

  return {
    visible,
    brightness: visibleCount ? brightnessSum / visibleCount : 0,
    visibleRatio: visibleCount / pixels,
    retinalColorScore: visibleCount
      ? (redSum / visibleCount) / Math.max(greenSum / visibleCount, blueSum / visibleCount, 1)
      : 0,
    sharpness,
  };
}

function majorityFilter(mask: Uint8Array, width: number, height: number) {
  const filtered = new Uint8Array(mask.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let count = 0;
      let samples = 0;
      for (let offsetY = -2; offsetY <= 2; offsetY += 1) {
        const sourceY = Math.min(height - 1, Math.max(0, y + offsetY));
        for (let offsetX = -2; offsetX <= 2; offsetX += 1) {
          const sourceX = Math.min(width - 1, Math.max(0, x + offsetX));
          count += mask[sourceY * width + sourceX];
          samples += 1;
        }
      }
      filtered[y * width + x] = count > samples / 2 ? 1 : 0;
    }
  }
  return filtered;
}

export async function preprocessImage(file: File): Promise<{ tensor: Float32Array; quality: ImageQuality }> {
  const bitmap = await createImageBitmap(file);
  const width = bitmap.width;
  const height = bitmap.height;
  if (!width || !height) {
    bitmap.close();
    throw new Error("Could not decode the uploaded image.");
  }

  const detectionScale = Math.min(1, DETECTION_MAX_SIDE / Math.max(width, height));
  const detectionWidth = Math.max(1, Math.round(width * detectionScale));
  const detectionHeight = Math.max(1, Math.round(height * detectionScale));
  const detectionCanvas = canvas(detectionWidth, detectionHeight);
  const detectionContext = detectionCanvas.getContext("2d", { willReadFrequently: true });
  if (!detectionContext) throw new Error("Image processing is unavailable in this browser.");
  detectionContext.drawImage(bitmap, 0, 0, detectionWidth, detectionHeight);
  const detectionData = detectionContext.getImageData(0, 0, detectionWidth, detectionHeight);
  const metrics = analyzePixels(detectionData.data, detectionWidth, detectionHeight);
  const component = largestComponent(
    majorityFilter(metrics.visible, detectionWidth, detectionHeight),
    detectionWidth,
    detectionHeight,
  );

  const warnings: string[] = [];
  const blocking: string[] = [];
  if (Math.min(width, height) < 224) blocking.push("Image resolution is below the 224 px minimum");
  if (metrics.visibleRatio < 0.15) blocking.push("Too little visible image area was detected");
  if (metrics.brightness < 20) blocking.push("Image is too dark for reliable processing");
  else if (metrics.brightness > 235) blocking.push("Image is too bright for reliable processing");
  if (metrics.sharpness < 2) warnings.push("Image may be blurred; interpret the result with extra caution");
  if (metrics.retinalColorScore < 1.05) warnings.push("Color profile is atypical for a retinal fundus photograph");

  const quality: ImageQuality = {
    width,
    height,
    brightness: metrics.brightness,
    sharpness: metrics.sharpness,
    visible_area_ratio: metrics.visibleRatio,
    retinal_color_score: metrics.retinalColorScore,
    acceptable: blocking.length === 0,
    warnings: [...blocking, ...warnings],
  };
  if (!quality.acceptable) {
    bitmap.close();
    throw new Error(blocking[0]);
  }

  let bounds: Bounds = { x: 0, y: 0, width, height };
  if (component) {
    const candidate = {
      x: Math.max(0, Math.floor(component.x / detectionScale)),
      y: Math.max(0, Math.floor(component.y / detectionScale)),
      width: Math.min(width, Math.ceil(component.width / detectionScale)),
      height: Math.min(height, Math.ceil(component.height / detectionScale)),
    };
    const retained = (candidate.width * candidate.height) / (width * height);
    if (retained >= 0.1) bounds = candidate;
  }

  const croppedCanvas = canvas(bounds.width, bounds.height);
  const croppedContext = croppedCanvas.getContext("2d", { willReadFrequently: true });
  if (!croppedContext) throw new Error("Image processing is unavailable in this browser.");
  croppedContext.drawImage(
    bitmap,
    bounds.x,
    bounds.y,
    bounds.width,
    bounds.height,
    0,
    0,
    bounds.width,
    bounds.height,
  );
  bitmap.close();

  const cropped = croppedContext.getImageData(0, 0, bounds.width, bounds.height);
  const centerX = Math.floor(bounds.width / 2);
  const centerY = Math.floor(bounds.height / 2);
  const radius = Math.round(Math.min(bounds.width, bounds.height) / 2);
  const radiusSquared = radius * radius;
  for (let y = 0; y < bounds.height; y += 1) {
    for (let x = 0; x < bounds.width; x += 1) {
      const dx = x - centerX;
      const dy = y - centerY;
      if (dx * dx + dy * dy <= radiusSquared) continue;
      const offset = (y * bounds.width + x) * 4;
      cropped.data[offset] = 0;
      cropped.data[offset + 1] = 0;
      cropped.data[offset + 2] = 0;
    }
  }
  croppedContext.putImageData(cropped, 0, 0);

  const resizedCanvas = canvas(INPUT_SIZE, INPUT_SIZE);
  const resizedContext = resizedCanvas.getContext("2d", { willReadFrequently: true });
  if (!resizedContext) throw new Error("Image processing is unavailable in this browser.");
  resizedContext.imageSmoothingEnabled = true;
  resizedContext.imageSmoothingQuality = "high";
  resizedContext.drawImage(croppedCanvas, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const resized = resizedContext.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;

  const plane = INPUT_SIZE * INPUT_SIZE;
  const tensor = new Float32Array(plane * 3);
  for (let index = 0; index < plane; index += 1) {
    const pixel = index * 4;
    tensor[index] = (resized[pixel] / 255 - MEAN[0]) / STD[0];
    tensor[plane + index] = (resized[pixel + 1] / 255 - MEAN[1]) / STD[1];
    tensor[plane * 2 + index] = (resized[pixel + 2] / 255 - MEAN[2]) / STD[2];
  }

  return { tensor, quality };
}
