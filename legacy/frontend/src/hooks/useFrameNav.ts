import { useCallback, useEffect, useRef, useState } from "react";
import { useAppStore } from "@/stores/appStore";

interface FrameNavOptions {
  /** Check if a frame is ready to display (e.g. cached). If not provided, always ready. */
  isFrameReady?: (idx: number) => boolean;
}

export function useFrameNav(options?: FrameNavOptions) {
  const numFrames = useAppStore((s) => s.numFrames);
  const currentFrame = useAppStore((s) => s.currentFrame);
  const setCurrentFrame = useAppStore((s) => s.setCurrentFrame);

  const playRef = useRef(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const fpsRef = useRef(5);
  const rafRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef(0);
  const isFrameReadyRef = useRef(options?.isFrameReady);

  // Keep ref in sync with latest callback
  useEffect(() => {
    isFrameReadyRef.current = options?.isFrameReady;
  }, [options?.isFrameReady]);

  const next = useCallback(() => {
    setCurrentFrame(Math.min(currentFrame + 1, numFrames - 1));
  }, [currentFrame, numFrames, setCurrentFrame]);

  const prev = useCallback(() => {
    setCurrentFrame(Math.max(currentFrame - 1, 0));
  }, [currentFrame, setCurrentFrame]);

  const first = useCallback(() => setCurrentFrame(0), [setCurrentFrame]);
  const last = useCallback(
    () => setCurrentFrame(numFrames - 1),
    [numFrames, setCurrentFrame]
  );

  const goTo = useCallback(
    (f: number) => setCurrentFrame(Math.max(0, Math.min(f, numFrames - 1))),
    [numFrames, setCurrentFrame]
  );

  const stopLoop = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);

  const pause = useCallback(() => {
    playRef.current = false;
    setIsPlaying(false);
    stopLoop();
  }, [stopLoop]);

  const play = useCallback(() => {
    if (playRef.current) return;

    // If at last frame, wrap around
    const state = useAppStore.getState();
    if (state.currentFrame >= state.numFrames - 1) {
      useAppStore.setState({ currentFrame: 0 });
    }

    playRef.current = true;
    setIsPlaying(true);
    lastFrameTimeRef.current = performance.now();

    const tick = (now: number) => {
      if (!playRef.current) return;

      const interval = 1000 / fpsRef.current;
      const elapsed = now - lastFrameTimeRef.current;

      if (elapsed >= interval) {
        const s = useAppStore.getState();
        const nextFrame = s.currentFrame + 1;

        if (nextFrame >= s.numFrames) {
          playRef.current = false;
          setIsPlaying(false);
          return;
        }

        // Only advance if frame is ready (or no readiness check)
        const ready = isFrameReadyRef.current
          ? isFrameReadyRef.current(nextFrame)
          : true;

        if (ready) {
          useAppStore.setState({ currentFrame: nextFrame });
          lastFrameTimeRef.current = now - (elapsed - interval); // drift correction
        }
        // If not ready, retry on next rAF — don't update lastFrameTime
      }

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
  }, []);

  const setFps = useCallback((fps: number) => {
    fpsRef.current = fps;
    // rAF loop reads fpsRef.current dynamically — no restart needed
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === "ArrowRight") next();
      else if (e.key === "ArrowLeft") prev();
      else if (e.key === " ") {
        e.preventDefault();
        playRef.current ? pause() : play();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [next, prev, play, pause]);

  // Cleanup on unmount
  useEffect(() => {
    return () => stopLoop();
  }, [stopLoop]);

  return {
    currentFrame, numFrames, next, prev, first, last, goTo,
    play, pause, isPlaying, setFps,
  };
}
