import { useState, useEffect, useRef } from "react";

/**
 * Synchronize two image sources so they appear together in the same paint.
 *
 * Preloads both images via hidden Image() objects before swapping the
 * displayed pair, preventing visual desync when one (e.g. a pre-cached
 * overlay blob) loads faster than the other (e.g. a network-fetched
 * background frame).
 *
 * When `enabled` is false, passes through sources immediately (no sync
 * overhead — used for reference-background mode where bg never changes).
 */
export function useSyncedImagePair(
  bgSrc: string,
  overlaySrc: string,
  enabled: boolean,
): { bg: string; overlay: string } {
  const [displayed, setDisplayed] = useState({ bg: bgSrc, overlay: overlaySrc });
  const pendingRef = useRef<object | null>(null);

  useEffect(() => {
    if (!enabled) {
      pendingRef.current = null;
      setDisplayed({ bg: bgSrc, overlay: overlaySrc });
      return;
    }

    // Token object — identity check detects superseded preloads
    // (handles rapid scrubbing: only the latest frame pair commits)
    const token = {};
    pendingRef.current = token;
    let bgReady = false;
    let overlayReady = false;

    const tryCommit = () => {
      if (pendingRef.current !== token) return;
      if (bgReady && overlayReady) {
        setDisplayed({ bg: bgSrc, overlay: overlaySrc });
        pendingRef.current = null;
      }
    };

    // Blob URLs (from pre-render cache) are already decoded in memory
    if (overlaySrc.startsWith("blob:")) {
      overlayReady = true;
    }

    const bgImg = new Image();
    bgImg.onload = bgImg.onerror = () => { bgReady = true; tryCommit(); };
    bgImg.src = bgSrc;

    if (!overlayReady) {
      const ovImg = new Image();
      ovImg.onload = ovImg.onerror = () => { overlayReady = true; tryCommit(); };
      ovImg.src = overlaySrc;
    } else {
      tryCommit();
    }
  }, [bgSrc, overlaySrc, enabled]);

  return displayed;
}
