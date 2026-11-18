/**
 * Client-side colormap application for data textures.
 *
 * Decodes 16-bit encoded data textures and applies colormap LUTs
 * to produce displayable RGBA images.
 */

export interface DataTexture {
  /** Raw RGBA ImageData from the decoded data texture PNG */
  imageData: ImageData;
  /** Minimum data value in the field */
  dataMin: number;
  /** Maximum data value in the field */
  dataMax: number;
}

/**
 * Decode a data texture PNG blob into a DataTexture.
 *
 * The PNG uses 16-bit encoding: R=high byte, G=low byte, A=valid mask.
 * Data range is carried in response headers X-Data-Min / X-Data-Max.
 */
export async function decodeDataTexture(
  blob: Blob,
  dataMin: number,
  dataMax: number,
): Promise<DataTexture> {
  const bmp = await createImageBitmap(blob);
  const canvas = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(bmp, 0, 0);
  bmp.close(); // release GPU/system memory
  const imageData = ctx.getImageData(0, 0, bmp.width, bmp.height);
  return { imageData, dataMin, dataMax };
}

/**
 * Apply a colormap LUT to a data texture, producing an RGBA blob URL
 * suitable for display in an <img> element.
 *
 * @param dt - Decoded data texture
 * @param lut - 1024-byte colormap LUT (256 × RGBA)
 * @param vmin - Display range minimum
 * @param vmax - Display range maximum
 * @param logScale - Whether to use logarithmic scaling
 * @returns Blob URL of the colorized RGBA image
 */
export async function colorizeDataTexture(
  dt: DataTexture,
  lut: Uint8Array,
  vmin: number,
  vmax: number,
  logScale: boolean,
): Promise<string> {
  const { imageData, dataMin, dataMax } = dt;
  const src = imageData.data;
  const w = imageData.width;
  const h = imageData.height;
  const out = new ImageData(w, h);
  const dst = out.data;

  const dataRange = dataMax - dataMin;

  // Pre-compute log scale constants
  const useLog = logScale && vmin > 0 && vmax > 0;
  const logVmin = useLog ? Math.log10(Math.max(vmin, 1e-20)) : 0;
  const logVmax = useLog ? Math.log10(Math.max(vmax, 1e-20)) : 0;
  const logRange = logVmax - logVmin;
  const valueRange = vmax - vmin;

  const pixelCount = w * h;
  for (let i = 0; i < pixelCount; i++) {
    const si = i * 4;
    const alpha = src[si + 3];

    if (alpha === 0) {
      // NaN pixel — transparent
      dst[si] = 0;
      dst[si + 1] = 0;
      dst[si + 2] = 0;
      dst[si + 3] = 0;
      continue;
    }

    // Decode 16-bit value: R=high byte, G=low byte
    const val16 = (src[si] << 8) | src[si + 1];
    const value = dataMin + (val16 / 65535) * dataRange;

    // Log scale: non-positive values are physically meaningless → transparent
    if (useLog && value <= 0) {
      dst[si] = dst[si + 1] = dst[si + 2] = dst[si + 3] = 0;
      continue;
    }

    // Map to [0, 1] within vmin/vmax range
    let fraction: number;
    if (useLog) {
      const logVal = Math.log10(value);
      fraction = logRange > 0 ? (logVal - logVmin) / logRange : 0.5;
    } else {
      fraction = valueRange > 1e-15 ? (value - vmin) / valueRange : 0.5;
    }

    // Clamp and look up in LUT
    const lutIdx =
      Math.max(0, Math.min(255, Math.round(fraction * 255))) << 2;
    dst[si] = lut[lutIdx];
    dst[si + 1] = lut[lutIdx + 1];
    dst[si + 2] = lut[lutIdx + 2];
    dst[si + 3] = 255; // fully opaque (CSS opacity handles transparency)
  }

  // Convert ImageData → blob URL
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext("2d")!;
  ctx.putImageData(out, 0, 0);
  const blob = await canvas.convertToBlob({ type: "image/png" });
  return URL.createObjectURL(blob);
}
