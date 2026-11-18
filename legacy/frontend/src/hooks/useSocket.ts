import { useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { getSocket } from "@/api/socket";
import { useAppStore } from "@/stores/appStore";
import { useToast } from "@/components/shared/Toast";

export function useSocket() {
  const setProcessing = useAppStore((s) => s.setProcessing);
  const setProcessingPaused = useAppStore((s) => s.setProcessingPaused);
  const setResults = useAppStore((s) => s.setResults);
  const setStrain = useAppStore((s) => s.setStrain);
  const setStrainComputing = useAppStore((s) => s.setStrainComputing);
  const setExport = useAppStore((s) => s.setExport);
  const navigate = useNavigate();
  const location = useLocation();
  const { toast } = useToast();

  useEffect(() => {
    const socket = getSocket();

    // Processing events
    socket.on("processing:progress", (d: { percent: number; current: number; total: number }) => {
      setProcessing(true, d.percent, d.current, d.total);
    });
    socket.on("processing:paused", () => {
      setProcessingPaused(true);
      toast("info", "Processing paused");
    });
    socket.on("processing:resumed", () => {
      setProcessingPaused(false);
      toast("info", "Processing resumed");
    });
    socket.on("processing:complete", (d: { stopped: boolean; num_frames: number }) => {
      setProcessing(false);
      if (d.num_frames > 0) {
        setResults(d.num_frames);
      }
      if (d.stopped) {
        toast("info", `Processing stopped — ${d.num_frames} frames computed`);
      } else {
        toast("success", `Processing complete — ${d.num_frames} frames`);
        // Auto-navigate to displacement page only on natural completion
        if (location.pathname === "/") {
          navigate("/displacement");
        }
      }
    });
    socket.on("processing:error", (d: { error: string }) => {
      setProcessing(false);
      toast("error", `Processing failed: ${d.error}`);
    });

    // Strain events
    socket.on("strain:progress", (d: { percent: number }) => {
      setStrainComputing(true, d.percent);
    });
    socket.on("strain:complete", (d: { num_frames: number; components: string[]; max_strain?: number; method?: string }) => {
      setStrain(true, d.components);
      toast("success", `Strain computed — ${d.components.length} components`);
      if (d.method === "engineering" && d.max_strain != null && d.max_strain > 0.05) {
        toast("warning",
          `Large deformation detected (max ${(d.max_strain * 100).toFixed(1)}%). ` +
          `Consider using Green-Lagrange strain for better accuracy.`
        );
      }
    });
    socket.on("strain:error", (d: { error: string }) => {
      setStrainComputing(false);
      toast("error", `Strain calculation failed: ${d.error}`);
    });

    // Export events
    socket.on("export:progress", (d: { percent: number }) => {
      setExport(true, d.percent);
    });
    socket.on("export:complete", () => {
      setExport(false, 100);
      toast("success", "Export complete");
    });
    socket.on("export:error", (d: { error: string }) => {
      setExport(false);
      toast("error", `Export failed: ${d.error}`);
    });

    // Connection events
    socket.on("connect", () => {
      toast("info", "Connected to server");
    });
    socket.on("disconnect", () => {
      toast("error", "Disconnected from server");
    });

    return () => {
      socket.off("processing:progress");
      socket.off("processing:paused");
      socket.off("processing:resumed");
      socket.off("processing:complete");
      socket.off("processing:error");
      socket.off("strain:progress");
      socket.off("strain:complete");
      socket.off("strain:error");
      socket.off("export:progress");
      socket.off("export:complete");
      socket.off("export:error");
      socket.off("connect");
      socket.off("disconnect");
    };
  }, [setProcessing, setProcessingPaused, setResults, setStrain, setStrainComputing, setExport, navigate, location.pathname, toast]);
}
