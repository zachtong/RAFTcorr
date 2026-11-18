import { useEffect, useCallback, useState } from "react";
import { Sidebar } from "@/components/layout/Sidebar";
import { VisualizationSettings } from "@/components/roi/VisualizationSettings";
import { DisplacementView } from "@/components/displacement/DisplacementView";
import { FramePlayback } from "@/components/displacement/FramePlayback";
import { ResultContextBanner } from "@/components/shared/ResultContextBanner";
import { usePreRenderCache } from "@/hooks/usePreRenderCache";
import { useAppStore } from "@/stores/appStore";

function useWindowViewport() {
  const [vp, setVp] = useState({ vw: window.innerWidth, vh: window.innerHeight });
  useEffect(() => {
    const h = () => setVp({ vw: window.innerWidth, vh: window.innerHeight });
    window.addEventListener("resize", h);
    return () => window.removeEventListener("resize", h);
  }, []);
  return vp;
}

export function DisplacementPage() {
  const vp = useWindowViewport();
  // Each panel is roughly half the window width
  const panelVp = { vw: Math.round(vp.vw / 2), vh: vp.vh };
  const cacheU = usePreRenderCache("u", panelVp);
  const cacheV = usePreRenderCache("v", panelVp);
  const hasResults = useAppStore((s) => s.hasResults);
  const numFrames = useAppStore((s) => s.numFrames);

  // Auto-start pre-render when results are available and there are multiple frames
  useEffect(() => {
    if (hasResults && numFrames > 1) {
      cacheU.startPreRender();
      cacheV.startPreRender();
    }
  }, [hasResults, numFrames, cacheU.startPreRender, cacheV.startPreRender]);

  const bothCached = cacheU.cachedCount >= cacheU.totalFrames && cacheV.cachedCount >= cacheV.totalFrames;
  const isFrameReady = useCallback(
    (idx: number) => {
      // If both frames are in pre-render cache, use them (instant)
      if (cacheU.getFrame(idx) && cacheV.getFrame(idx)) return true;
      // If not all frames are cached yet, advance anyway —
      // DisplacementView will fall back to the live server URL
      return !bothCached;
    },
    [cacheU.getFrame, cacheV.getFrame, bothCached]
  );

  const combinedProgress = Math.round(
    ((cacheU.cachedCount + cacheV.cachedCount) /
      (cacheU.totalFrames + cacheV.totalFrames || 1)) *
      100
  );

  return (
    <div className="flex flex-1 overflow-hidden">
      <Sidebar>
        <div className="flex-1 overflow-y-auto">
          <VisualizationSettings />
        </div>
      </Sidebar>
      <div className="flex-1 flex flex-col overflow-hidden">
        <ResultContextBanner />
        <DisplacementView cacheU={cacheU} cacheV={cacheV} />
        <FramePlayback
          preRenderCache={{
            isPreRendering: cacheU.isPreRendering || cacheV.isPreRendering,
            progress: combinedProgress,
            cachedCount: cacheU.cachedCount + cacheV.cachedCount,
            totalFrames: cacheU.totalFrames + cacheV.totalFrames,
            isFrameReady,
          }}
        />
      </div>
    </div>
  );
}
