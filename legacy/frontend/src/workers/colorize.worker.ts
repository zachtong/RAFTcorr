/**
 * Web Worker for off-main-thread colormap application.
 *
 * Receives a data texture (ImageData buffer) + colormap LUT + display range,
 * returns a colorized RGBA PNG blob.
 *
 * Using a worker avoids blocking the main thread during pixel-level
 * colorization of large displacement/strain fields.
 */

export interface ColorizeRequest {
  /** Unique request ID for correlating responses */
  id: number;
  /** Raw RGBA pixel data from the data texture PNG (Uint8ClampedArray buffer) */
  imageBuffer: ArrayBuffer;
  width: number;
  height: number;
  /** 1024-byte colormap LUT (256 x RGBA) */
  lutBuffer: ArrayBuffer;
  /** Data value range encoded in the texture */
  dataMin: number;
  dataMax: number;
  /** Display range */
  vmin: number;
  vmax: number;
  logScale: boolean;
}

export interface ColorizeResponse {
  id: number;
  /** Colorized PNG as a Blob, or null on error */
  blob: Blob | null;
}

self.onmessage = async (e: MessageEvent<ColorizeRequest>) => {
  const {
    id,
    imageBuffer,
    width,
    height,
    lutBuffer,
    dataMin,
    dataMax,
    vmin,
    vmax,
    logScale,
  } = e.data;

  try {
    const src = new Uint8ClampedArray(imageBuffer);
    const lut = new Uint8Array(lutBuffer);
    const out = new Uint8ClampedArray(width * height * 4);

    const dataRange = dataMax - dataMin;
    const useLog = logScale && vmin > 0 && vmax > 0;
    const logVmin = useLog ? Math.log10(Math.max(vmin, 1e-20)) : 0;
    const logVmax = useLog ? Math.log10(Math.max(vmax, 1e-20)) : 0;
    const logRange = logVmax - logVmin;
    const valueRange = vmax - vmin;

    const pixelCount = width * height;
    for (let i = 0; i < pixelCount; i++) {
      const si = i * 4;
      const alpha = src[si + 3];

      if (alpha === 0) {
        // NaN pixel — transparent
        out[si] = 0;
        out[si + 1] = 0;
        out[si + 2] = 0;
        out[si + 3] = 0;
        continue;
      }

      // Decode 16-bit value: R=high byte, G=low byte
      const val16 = (src[si] << 8) | src[si + 1];
      const value = dataMin + (val16 / 65535) * dataRange;

      if (useLog && value <= 0) {
        out[si] = out[si + 1] = out[si + 2] = out[si + 3] = 0;
        continue;
      }

      let fraction: number;
      if (useLog) {
        const logVal = Math.log10(value);
        fraction = logRange > 0 ? (logVal - logVmin) / logRange : 0.5;
      } else {
        fraction = valueRange > 1e-15 ? (value - vmin) / valueRange : 0.5;
      }

      const lutIdx =
        Math.max(0, Math.min(255, Math.round(fraction * 255))) << 2;
      out[si] = lut[lutIdx];
      out[si + 1] = lut[lutIdx + 1];
      out[si + 2] = lut[lutIdx + 2];
      out[si + 3] = 255;
    }

    // Convert to PNG blob via OffscreenCanvas
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext("2d")!;
    ctx.putImageData(new ImageData(out, width, height), 0, 0);
    const blob = await canvas.convertToBlob({ type: "image/png" });

    const response: ColorizeResponse = { id, blob };
    self.postMessage(response);
  } catch {
    const response: ColorizeResponse = { id, blob: null };
    self.postMessage(response);
  }
};
