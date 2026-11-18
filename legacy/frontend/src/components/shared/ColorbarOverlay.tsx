import type { ColorbarSettings } from "@/stores/appStore";

// Raw color stops per colormap — used to build both continuous and discrete gradients
const COLORMAP_STOPS: Record<string, string[]> = {
  turbo: [
    "#30123b", "#4662d7", "#36aaf9", "#1ae4b6", "#72fe5e",
    "#c8ef34", "#faba39", "#f66b19", "#ca2a04", "#7a0403",
  ],
  viridis: [
    "#440154", "#482878", "#3e4989", "#31688e", "#26828e",
    "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725",
  ],
  jet: ["#0000ff", "#00ffff", "#00ff00", "#ffff00", "#ff8800", "#ff0000"],
  coolwarm: [
    "#3b4cc0", "#6788ee", "#9abbff", "#c9d7ef",
    "#edd1c2", "#f7a789", "#e26952", "#b40426",
  ],
  plasma: [
    "#0d0887", "#5b02a3", "#9c179e", "#cb4679",
    "#ed7953", "#fdb42f", "#f0f921",
  ],
  inferno: [
    "#000004", "#1b0c41", "#4a0c6b", "#781c6d", "#a52c60",
    "#cf4446", "#ed6925", "#fb9b06", "#f7d13d", "#fcffa4",
  ],
};

// Human-readable component labels (mirrors COMPONENT_LABELS in export_animation.py)
export const COMPONENT_LABELS: Record<string, string> = {
  u: "U displacement",
  v: "V displacement",
  magnitude: "Magnitude",
  velocity: "Velocity",
  exx: "εxx",
  eyy: "εyy",
  exy: "εxy",
  e1: "ε1",
  e2: "ε2",
  max_shear: "Max Shear",
  von_mises: "Von Mises",
  rotation: "Rotation",
  dexx_dt: "dεxx/dt",
  deyy_dt: "dεyy/dt",
  dexy_dt: "dεxy/dt",
};

// --- Color interpolation utilities ---

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.substring(0, 2), 16),
    parseInt(h.substring(2, 4), 16),
    parseInt(h.substring(4, 6), 16),
  ];
}

function rgbToHex(r: number, g: number, b: number): string {
  return (
    "#" +
    [r, g, b]
      .map((c) => Math.round(Math.max(0, Math.min(255, c))).toString(16).padStart(2, "0"))
      .join("")
  );
}

/** Sample a color at position t (0-1) from an array of color stops */
function sampleColor(stops: string[], t: number): string {
  const clamped = Math.max(0, Math.min(1, t));
  const idx = clamped * (stops.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return stops[lo];
  const frac = idx - lo;
  const [r1, g1, b1] = hexToRgb(stops[lo]);
  const [r2, g2, b2] = hexToRgb(stops[hi]);
  return rgbToHex(
    r1 + (r2 - r1) * frac,
    g1 + (g2 - g1) * frac,
    b1 + (b2 - b1) * frac
  );
}

/** Build a CSS gradient string — continuous or discrete (stepped) */
function buildGradient(colormap: string, discreteLevels: number): string {
  const stops = COLORMAP_STOPS[colormap] ?? COLORMAP_STOPS.turbo;

  if (discreteLevels <= 0) {
    // Continuous gradient
    return `linear-gradient(to top, ${stops.join(", ")})`;
  }

  // Discrete: sample N colors, create stepped gradient
  const n = discreteLevels;
  const parts: string[] = [];
  for (let i = 0; i < n; i++) {
    const t = n === 1 ? 0.5 : i / (n - 1);
    const color = sampleColor(stops, t);
    const start = ((i / n) * 100).toFixed(1);
    const end = (((i + 1) / n) * 100).toFixed(1);
    parts.push(`${color} ${start}% ${end}%`);
  }
  return `linear-gradient(to top, ${parts.join(", ")})`;
}

// --- Formatting & tick generation ---

function fmt(v: number): string {
  const abs = Math.abs(v);
  if (abs === 0) return "0";
  if (abs >= 1000 || abs < 0.01) return v.toExponential(1);
  return parseFloat(v.toPrecision(3)).toString();
}

function linearTicks(vmin: number, vmax: number, count: number): number[] {
  const n = Math.max(2, count);
  const ticks: number[] = [];
  for (let i = 0; i < n; i++) ticks.push(vmin + (vmax - vmin) * (i / (n - 1)));
  return ticks;
}

function logTicks(vmin: number, vmax: number): number[] {
  const safeMin = Math.max(vmin, 1e-20);
  const safeMax = Math.max(vmax, 1e-20);
  const logMin = Math.log10(safeMin);
  const logMax = Math.log10(safeMax);

  const ticks: number[] = [vmin];
  const minPow = Math.ceil(logMin);
  const maxPow = Math.floor(logMax);
  for (let p = minPow; p <= maxPow; p++) {
    const val = Math.pow(10, p);
    if (val > safeMin * 1.01 && val < safeMax * 0.99) ticks.push(val);
  }
  ticks.push(vmax);

  if (ticks.length < 4) {
    const extra: number[] = [];
    for (let i = 1; i <= 3; i++) {
      const logVal = logMin + (logMax - logMin) * (i / 4);
      extra.push(Math.pow(10, logVal));
    }
    ticks.splice(1, 0, ...extra);
  }
  return ticks;
}

function tickPct(value: number, vmin: number, vmax: number, isLog: boolean): number {
  if (isLog) {
    const logMin = Math.log10(Math.max(vmin, 1e-20));
    const logMax = Math.log10(Math.max(vmax, 1e-20));
    if (logMax === logMin) return 50;
    return ((Math.log10(Math.max(value, 1e-20)) - logMin) / (logMax - logMin)) * 100;
  }
  if (vmax === vmin) return 50;
  return ((value - vmin) / (vmax - vmin)) * 100;
}

// --- Component ---

interface ColorbarOverlayProps {
  colormap: string;
  vmin?: number;
  vmax?: number;
  unit?: string;
  scaleFactor?: number;
  logScale?: boolean;
  component?: string;
  colorbarSettings?: ColorbarSettings;
}

export function ColorbarOverlay({
  colormap,
  vmin,
  vmax,
  unit,
  scaleFactor = 1,
  logScale = false,
  component,
  colorbarSettings,
}: ColorbarOverlayProps) {
  const hasRange = vmin != null && vmax != null;

  // Extract settings with defaults
  const labelText = colorbarSettings?.labelText || "";
  const labelFontSize = colorbarSettings?.labelFontSize ?? 12;
  const tickCount = colorbarSettings?.tickCount || 5; // 0 → default 5
  const tickFontSize = colorbarSettings?.tickFontSize ?? 10;
  const barThickness = colorbarSettings?.barThickness ?? 1.0;
  const hideOutline = colorbarSettings?.hideOutline ?? false;
  const discreteLevels = colorbarSettings?.discreteLevels ?? 0;

  // Resolve display label: custom > auto from component
  const displayLabel = labelText || (component ? COMPONENT_LABELS[component] ?? component : "");

  const ticks = hasRange
    ? logScale
      ? logTicks(vmin, vmax)
      : linearTicks(vmin, vmax, tickCount)
    : [];

  const gradient = buildGradient(colormap, discreteLevels);

  return (
    <div className="absolute right-3 top-8 bottom-8 z-10 pointer-events-none flex flex-col bg-black/40 backdrop-blur-sm rounded-md p-1.5 gap-1">
      {/* Label at top */}
      {displayLabel && (
        <span
          className="text-white/90 font-sans leading-none text-center shrink-0 max-w-[5rem] truncate"
          style={{ fontSize: `${Math.max(7, labelFontSize - 2)}px` }}
          title={displayLabel}
        >
          {displayLabel}
        </span>
      )}
      {/* Unit label */}
      {unit && (
        <span className="text-[9px] text-white/70 font-mono leading-none text-center shrink-0">
          [{unit}]
        </span>
      )}
      {/* Gradient bar + tick labels */}
      <div className="flex-1 flex gap-1 min-h-0">
        {/* Gradient bar with tick marks — width controlled by barThickness */}
        <div
          className={`relative rounded-sm shrink-0 ${hideOutline ? "" : "border border-white/25"}`}
          style={{ background: gradient, width: `${Math.round(16 * barThickness)}px` }}
        >
          {hasRange &&
            ticks.map((val, i) => {
              const pct = tickPct(val, vmin, vmax, logScale);
              return (
                <div
                  key={i}
                  className="absolute right-0 w-1.5 border-t border-white/80"
                  style={{ bottom: `${pct}%` }}
                />
              );
            })}
        </div>
        {/* Tick value labels */}
        <div className="relative min-w-[2rem]">
          {hasRange &&
            ticks.map((val, i) => {
              const pct = tickPct(val, vmin, vmax, logScale);
              return (
                <span
                  key={i}
                  className="absolute left-0 text-white font-mono leading-none whitespace-nowrap"
                  style={{
                    fontSize: `${tickFontSize}px`,
                    bottom: `${pct}%`,
                    transform: "translateY(50%)",
                  }}
                >
                  {fmt(val * scaleFactor)}
                </span>
              );
            })}
        </div>
      </div>
    </div>
  );
}
