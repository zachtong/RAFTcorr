/**
 * Pool of colorize web workers for off-main-thread colormap application.
 *
 * Provides a simple async API: `colorize(dt, lut, vmin, vmax, logScale) → blobUrl`
 * that distributes work across a fixed number of workers.
 */

import type { DataTexture } from "@/lib/applyColormap";
import type { ColorizeRequest, ColorizeResponse } from "./colorize.worker";

const POOL_SIZE = 2;

interface PendingRequest {
  resolve: (url: string | null) => void;
}

let workers: Worker[] = [];
let nextRequestId = 0;
const pending = new Map<number, PendingRequest>();
let nextWorkerIdx = 0;

function ensurePool(): Worker[] {
  if (workers.length > 0) return workers;

  workers = Array.from({ length: POOL_SIZE }, () => {
    const w = new Worker(
      new URL("./colorize.worker.ts", import.meta.url),
      { type: "module" },
    );
    w.onmessage = (e: MessageEvent<ColorizeResponse>) => {
      const { id, blob } = e.data;
      const req = pending.get(id);
      if (req) {
        pending.delete(id);
        if (blob) {
          req.resolve(URL.createObjectURL(blob));
        } else {
          req.resolve(null);
        }
      }
    };
    return w;
  });

  return workers;
}

/**
 * Colorize a data texture using the worker pool.
 *
 * Copies the ImageData buffer to avoid transferring ownership
 * (the DataTexture may be reused for re-colorization).
 */
export function colorizeInWorker(
  dt: DataTexture,
  lut: Uint8Array,
  vmin: number,
  vmax: number,
  logScale: boolean,
): Promise<string | null> {
  const pool = ensurePool();
  const id = nextRequestId++;

  // Copy buffers (don't transfer — dt is reused across color changes)
  const imageBuffer = dt.imageData.data.buffer.slice(0) as ArrayBuffer;
  const lutBuffer = lut.buffer.slice(0) as ArrayBuffer;

  const msg: ColorizeRequest = {
    id,
    imageBuffer,
    width: dt.imageData.width,
    height: dt.imageData.height,
    lutBuffer,
    dataMin: dt.dataMin,
    dataMax: dt.dataMax,
    vmin,
    vmax,
    logScale,
  };

  return new Promise((resolve) => {
    pending.set(id, { resolve });

    // Round-robin across workers
    const worker = pool[nextWorkerIdx % pool.length];
    nextWorkerIdx++;

    worker.postMessage(msg, [imageBuffer, lutBuffer]);
  });
}

/**
 * Terminate all workers in the pool.
 * Call this on app unmount or when the cache is fully invalidated.
 */
export function terminatePool(): void {
  for (const w of workers) w.terminate();
  workers = [];
  // Reject any pending requests
  for (const [, req] of pending) req.resolve(null);
  pending.clear();
}
