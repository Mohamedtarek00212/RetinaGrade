function interpolate(left: number, right: number, amount: number) {
  return Math.round(left + (right - left) * amount);
}

function heatColor(value: number) {
  const bounded = Math.min(1, Math.max(0, value));
  if (bounded < 0.5) {
    const amount = bounded * 2;
    return [
      interpolate(246, 232, amount),
      interpolate(200, 106, amount),
      interpolate(95, 51, amount),
    ];
  }
  const amount = (bounded - 0.5) * 2;
  return [
    interpolate(232, 185, amount),
    interpolate(106, 58, amount),
    interpolate(51, 67, amount),
  ];
}

export function renderActivationOverlay(
  source: HTMLCanvasElement,
  values: Float32Array,
  width: number,
  height: number,
) {
  const heatmap = document.createElement("canvas");
  heatmap.width = width;
  heatmap.height = height;
  const heatmapContext = heatmap.getContext("2d");
  if (!heatmapContext) throw new Error("Explanation rendering is unavailable in this browser.");

  const pixels = heatmapContext.createImageData(width, height);
  for (let index = 0; index < values.length; index += 1) {
    const value = Math.min(1, Math.max(0, values[index]));
    const [red, green, blue] = heatColor(value);
    const offset = index * 4;
    pixels.data[offset] = red;
    pixels.data[offset + 1] = green;
    pixels.data[offset + 2] = blue;
    pixels.data[offset + 3] = value < 0.12 ? 0 : Math.round(40 + value * 155);
  }
  heatmapContext.putImageData(pixels, 0, 0);

  const overlay = document.createElement("canvas");
  overlay.width = source.width;
  overlay.height = source.height;
  const overlayContext = overlay.getContext("2d");
  if (!overlayContext) throw new Error("Explanation rendering is unavailable in this browser.");
  overlayContext.drawImage(source, 0, 0);
  overlayContext.imageSmoothingEnabled = true;
  overlayContext.imageSmoothingQuality = "high";
  overlayContext.drawImage(heatmap, 0, 0, overlay.width, overlay.height);
  return overlay.toDataURL("image/jpeg", 0.92);
}
