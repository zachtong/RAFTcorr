import { useEffect, useState, useMemo, useRef, useCallback } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import { useAppStore } from "@/stores/appStore";
import { getTimeSeries, getExtensometer, downloadProbeCSV } from "@/api/probes";
import { useToast } from "@/components/shared/Toast";
import { Download, Camera } from "lucide-react";

interface ChartPoint {
  frame: number;
  [key: string]: number | null;
}

export function TimeSeriesChart() {
  const probes = useAppStore((s) => s.probes);
  const currentFrame = useAppStore((s) => s.currentFrame);
  const displayComponent = useAppStore((s) => s.displayComponent);
  const setCurrentFrame = useAppStore((s) => s.setCurrentFrame);
  const activeProbeTab = useAppStore((s) => s.activeProbeTab);
  const [data, setData] = useState<ChartPoint[]>([]);
  const [lineMode, setLineMode] = useState<"value" | "strain">("value");
  const [metric, setMetric] = useState<"avg" | "max" | "min">("avg");
  const { toast } = useToast();
  const chartRef = useRef<HTMLDivElement>(null);

  const pointProbes = probes.filter((p) => p.type === "point");
  const lineProbes = probes.filter((p) => p.type === "line");
  const areaProbes = probes.filter((p) => p.type === "area");

  // Build a color lookup: probe id → color
  const probeColorMap = useMemo(() => {
    const map: Record<string, string> = {};
    for (const p of probes) {
      map[String(p.id)] = p.color;
    }
    return map;
  }, [probes]);

  // Determine what probes to show based on the active tab
  const showPointData = activeProbeTab === "point" && pointProbes.length > 0;
  const showLineData = activeProbeTab === "line" && lineProbes.length > 0;
  const showAreaData = activeProbeTab === "area" && areaProbes.length > 0;
  const hasData = showPointData || showLineData || showAreaData;

  // --- Download handlers ---
  const handleDownloadCSV = useCallback(async () => {
    try {
      const probeType = showPointData ? "point" : showLineData ? "line" : "area";
      const includeExt = showLineData && lineMode === "strain";
      await downloadProbeCSV(displayComponent, probeType, metric, includeExt);
      toast("success", "CSV downloaded");
    } catch {
      toast("error", "CSV download failed");
    }
  }, [showPointData, showLineData, displayComponent, metric, lineMode, toast]);

  const handleSaveChartPNG = useCallback(async () => {
    const container = chartRef.current;
    if (!container) return;
    const svgElement = container.querySelector(".recharts-wrapper svg");
    if (!svgElement) {
      toast("error", "Chart not found");
      return;
    }

    try {
      const svgClone = svgElement.cloneNode(true) as SVGSVGElement;
      // Inline computed styles for CSS variable resolution
      const allElements = svgClone.querySelectorAll("*");
      const origElements = svgElement.querySelectorAll("*");
      for (let i = 0; i < allElements.length; i++) {
        const computed = getComputedStyle(origElements[i]);
        const el = allElements[i] as SVGElement;
        if (computed.fill) el.style.fill = computed.fill;
        if (computed.stroke) el.style.stroke = computed.stroke;
        if (computed.fontSize) el.style.fontSize = computed.fontSize;
        if (computed.fontFamily) el.style.fontFamily = computed.fontFamily;
      }

      const svgData = new XMLSerializer().serializeToString(svgClone);
      const svgBlob = new Blob([svgData], { type: "image/svg+xml;charset=utf-8" });
      const url = URL.createObjectURL(svgBlob);

      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement("canvas");
        const scale = 2;
        canvas.width = img.width * scale;
        canvas.height = img.height * scale;
        const ctx = canvas.getContext("2d")!;
        ctx.scale(scale, scale);
        // Fill background
        const bgColor = getComputedStyle(document.documentElement)
          .getPropertyValue("--card").trim() || "#1e1e2e";
        ctx.fillStyle = bgColor;
        ctx.fillRect(0, 0, img.width, img.height);
        ctx.drawImage(img, 0, 0);

        canvas.toBlob((blob) => {
          if (!blob) return;
          const a = document.createElement("a");
          a.href = URL.createObjectURL(blob);
          a.download = `chart_${displayComponent}.png`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
          toast("success", "Chart saved as PNG");
        }, "image/png");
      };
      img.src = url;
    } catch {
      toast("error", "Failed to save chart");
    }
  }, [displayComponent, toast]);

  // Fetch time series data based on active tab
  useEffect(() => {
    if (!hasData) {
      setData([]);
      return;
    }

    let cancelled = false;

    const fetchData = async () => {
      try {
        if (showLineData && lineMode === "strain") {
          // Extensometer: engineering strain for each line probe
          const results = await Promise.all(
            lineProbes.map((p) => getExtensometer(p.id))
          );
          if (cancelled) return;

          const frames = results[0]?.num_frames ?? 0;
          const chartData: ChartPoint[] = [];
          for (let i = 0; i < frames; i++) {
            const point: ChartPoint = { frame: i + 1 };
            results.forEach((r, idx) => {
              point[`line_${lineProbes[idx].id}`] = r.strains[i];
            });
            chartData.push(point);
          }
          setData(chartData);
        } else {
          // Component value time series (point, line value mode, or area)
          const probeType = showPointData ? "point" : showLineData ? "line" : "area";
          const keyPrefix = showAreaData ? "area" : "probe";
          const result = await getTimeSeries(displayComponent, probeType, metric);
          if (cancelled) return;

          const frames = Object.values(result.series)[0]?.length ?? 0;
          const chartData: ChartPoint[] = [];
          for (let i = 0; i < frames; i++) {
            const point: ChartPoint = { frame: i + 1 };
            for (const [id, values] of Object.entries(result.series)) {
              point[`${keyPrefix}_${id}`] = values[i];
            }
            chartData.push(point);
          }
          setData(chartData);
        }
      } catch (e) {
        console.error("Failed to fetch time series:", e);
      }
    };

    fetchData();
    return () => { cancelled = true; };
  }, [hasData, showPointData, showLineData, showAreaData, pointProbes.length, lineProbes.length, areaProbes.length, displayComponent, lineMode, metric]);

  if (!hasData || data.length === 0) return null;

  const seriesKeys = Object.keys(data[0] || {}).filter((k) => k !== "frame");

  // Resolve the color for a series key like "probe_3", "line_2", or "area_1"
  const getColorForKey = (key: string): string => {
    const match = key.match(/(?:probe|line|area)_(\d+)/);
    if (match) {
      const id = match[1];
      if (probeColorMap[id]) return probeColorMap[id];
    }
    return "#888";
  };

  const title = showLineData && lineMode === "strain"
    ? "Engineering Strain over Time"
    : showAreaData
      ? `Area ${displayComponent.toUpperCase()} over Time`
      : `${displayComponent.toUpperCase()} over Time`;

  return (
    <div ref={chartRef} className="h-full border-t border-[var(--border)] bg-[var(--card)] px-2 pt-1 pb-2">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[10px] text-[var(--muted-foreground)] uppercase tracking-wider font-semibold">
          {title}
        </span>
        <div className="flex items-center gap-1">
          {/* Line tab: toggle between value & strain modes */}
          {showLineData && (
            <>
              <button
                onClick={() => setLineMode("value")}
                className={`px-2 py-0.5 rounded text-[9px] ${
                  lineMode === "value"
                    ? "bg-[var(--primary)] text-white"
                    : "text-[var(--muted-foreground)] hover:bg-[var(--secondary)]"
                }`}
              >
                Value
              </button>
              <button
                onClick={() => setLineMode("strain")}
                className={`px-2 py-0.5 rounded text-[9px] ${
                  lineMode === "strain"
                    ? "bg-[var(--primary)] text-white"
                    : "text-[var(--muted-foreground)] hover:bg-[var(--secondary)]"
                }`}
              >
                Strain
              </button>
            </>
          )}
          {/* Metric selector: for line value mode or area probes */}
          {((showLineData && lineMode === "value") || showAreaData) && (
            <select
              value={metric}
              onChange={(e) => setMetric(e.target.value as "avg" | "max" | "min")}
              className="px-1.5 py-0.5 rounded text-[9px] bg-[var(--input)] border border-[var(--border)] text-[var(--foreground)]"
            >
              <option value="avg">Avg</option>
              <option value="max">Max</option>
              <option value="min">Min</option>
            </select>
          )}
          {/* Separator */}
          <div className="w-px h-3 bg-[var(--border)] mx-0.5" />
          {/* CSV download */}
          <button
            onClick={handleDownloadCSV}
            className="p-0.5 rounded hover:bg-[var(--secondary)] text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
            title="Download CSV"
          >
            <Download size={12} />
          </button>
          {/* Chart PNG save */}
          <button
            onClick={handleSaveChartPNG}
            className="p-0.5 rounded hover:bg-[var(--secondary)] text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
            title="Save chart as PNG"
          >
            <Camera size={12} />
          </button>
        </div>
      </div>
      <ResponsiveContainer width="100%" height="85%">
        <LineChart
          data={data}
          onClick={(e: any) => {
            if (e?.activeLabel) setCurrentFrame(Number(e.activeLabel) - 1);
          }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="frame"
            tick={{ fontSize: 9, fill: "var(--muted-foreground)" }}
            axisLine={{ stroke: "var(--border)" }}
          />
          <YAxis
            tick={{ fontSize: 9, fill: "var(--muted-foreground)" }}
            axisLine={{ stroke: "var(--border)" }}
            width={50}
          />
          <Tooltip
            contentStyle={{
              background: "var(--card)",
              border: "1px solid var(--border)",
              borderRadius: "4px",
              fontSize: "10px",
            }}
          />
          <ReferenceLine
            x={currentFrame + 1}
            stroke="var(--primary)"
            strokeDasharray="3 3"
          />
          {seriesKeys.map((key) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={getColorForKey(key)}
              dot={false}
              strokeWidth={1.5}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
