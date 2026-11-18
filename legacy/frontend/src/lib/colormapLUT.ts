/**
 * Client-side colormap LUT management.
 *
 * Fetches colormap lookup tables from the server as 256×1 RGBA PNGs,
 * decodes them into Uint8Array LUTs, and caches them in memory.
 */

const lutCache = new Map<string, Uint8Array>();
const pending = new Map<string, Promise<Uint8Array>>();

/**
 * Fetch and cache a colormap LUT. Returns a 1024-byte Uint8Array
 * (256 entries × 4 channels: R, G, B, A).
 */
export async function getColormapLUT(name: string): Promise<Uint8Array> {
  const cached = lutCache.get(name);
  if (cached) return cached;

  // Deduplicate concurrent fetches for the same colormap
  const inflight = pending.get(name);
  if (inflight) return inflight;

  const promise = (async () => {
    const resp = await fetch(`/api/displacement/colormap/${name}`);
    if (!resp.ok) throw new Error(`Failed to fetch colormap: ${name}`);
    const blob = await resp.blob();
    const bmp = await createImageBitmap(blob);
    const canvas = new OffscreenCanvas(256, 1);
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(bmp, 0, 0);
    bmp.close(); // release GPU/system memory
    const imageData = ctx.getImageData(0, 0, 256, 1);
    const lut = new Uint8Array(imageData.data.buffer);
    lutCache.set(name, lut);
    pending.delete(name);
    return lut;
  })();

  pending.set(name, promise);
  return promise;
}

/**
 * Pre-warm a colormap LUT so it's ready when needed.
 */
export function prefetchColormapLUT(name: string): void {
  if (!lutCache.has(name)) {
    getColormapLUT(name).catch(() => {});
  }
}
