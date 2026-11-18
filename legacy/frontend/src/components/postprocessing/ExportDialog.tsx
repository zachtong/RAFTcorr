import { useState, useEffect, useRef } from "react";
import { CollapsibleSection } from "@/components/shared/CollapsibleSection";
import { SmallInput } from "@/components/shared/SmallInput";
import { FileBrowser } from "@/components/shared/FileBrowser";
import { useAppStore } from "@/stores/appStore";
import { useToast } from "@/components/shared/Toast";
import {
  exportScientific, exportImages, cancelExport,
  exportAnimation, exportReport,
} from "@/api/export";
import { Download, FolderOpen, Film, FileText } from "lucide-react";
import type { DisplayComponent } from "@/types/api";

const DISPLACEMENT_COMPONENTS: { value: DisplayComponent; label: string }[] = [
  { value: "u", label: "U (horizontal)" },
  { value: "v", label: "V (vertical)" },
  { value: "magnitude", label: "Magnitude" },
  { value: "velocity", label: "Velocity" },
];

const STRAIN_COMPONENTS: { value: DisplayComponent; label: string }[] = [
  { value: "exx", label: "εxx" },
  { value: "eyy", label: "εyy" },
  { value: "exy", label: "εxy" },
  { value: "e1", label: "ε1" },
  { value: "e2", label: "ε2" },
  { value: "max_shear", label: "Max Shear" },
  { value: "von_mises", label: "Von Mises" },
  { value: "rotation", label: "Rotation" },
  { value: "dexx_dt", label: "dεxx/dt" },
  { value: "deyy_dt", label: "dεyy/dt" },
  { value: "dexy_dt", label: "dεxy/dt" },
];

interface CompEntry {
  enabled: boolean;
  vmin: string;
  vmax: string;
  auto: boolean;
}

const inputClass =
  "w-full h-6 bg-[var(--input)] border border-[#3a3d45] rounded px-1.5 text-[10px] text-[var(--foreground)] focus:border-[var(--primary)] focus:outline-none";

const btnClass =
  "w-full flex items-center justify-center gap-1 py-1.5 bg-[var(--secondary)] hover:bg-[var(--secondary)]/80 rounded text-[11px] text-[var(--foreground)] disabled:opacity-40";

const SCI_VAR_GROUPS: { title: string; keys: { id: string; label: string }[] }[] = [
  {
    title: "Coordinates",
    keys: [
      { id: "X_ref", label: "X_ref" },
      { id: "Y_ref", label: "Y_ref" },
      { id: "ROI_mask", label: "ROI_mask" },
    ],
  },
  {
    title: "Displacement",
    keys: [
      { id: "U", label: "U" },
      { id: "V", label: "V" },
      { id: "Magnitude", label: "Magnitude" },
      { id: "Velocity", label: "Velocity" },
    ],
  },
  {
    title: "Strain",
    keys: [
      { id: "exx", label: "εxx" },
      { id: "eyy", label: "εyy" },
      { id: "exy", label: "εxy" },
      { id: "e1", label: "ε1" },
      { id: "e2", label: "ε2" },
      { id: "max_shear", label: "Max Shear" },
      { id: "von_mises", label: "Von Mises" },
      { id: "rotation", label: "Rotation" },
    ],
  },
];

interface VariablePickerProps {
  vars: Record<string, boolean>;
  onToggle: (k: string) => void;
  onAll: () => void;
  onNone: () => void;
  /** Hide the Strain group when no strain has been computed. */
  hasStrain: boolean;
}

function ScientificVariablePicker({
  vars, onToggle, onAll, onNone, hasStrain,
}: VariablePickerProps) {
  const visibleGroups = SCI_VAR_GROUPS.filter(
    (g) => g.title !== "Strain" || hasStrain,
  );
  return (
    <div className="border border-[var(--border)] rounded p-1.5 space-y-1 bg-[var(--input)]/40">
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-medium text-[var(--muted-foreground)] uppercase tracking-wide">
          Variables to export
        </span>
        <div className="flex gap-1">
          <button
            type="button"
            onClick={onAll}
            className="text-[9px] px-1.5 py-0.5 rounded bg-[var(--secondary)] hover:bg-[var(--secondary)]/80 text-[var(--foreground)]"
            title="Select all variables"
          >
            All
          </button>
          <button
            type="button"
            onClick={onNone}
            className="text-[9px] px-1.5 py-0.5 rounded bg-[var(--secondary)] hover:bg-[var(--secondary)]/80 text-[var(--foreground)]"
            title="Clear all variables"
          >
            None
          </button>
        </div>
      </div>
      {visibleGroups.map((group) => (
        <div key={group.title} className="space-y-0.5">
          <span className="text-[9px] text-[var(--muted-foreground)]">
            {group.title}
          </span>
          <div className="grid grid-cols-3 gap-x-2 gap-y-0.5 pl-1">
            {group.keys.map(({ id, label }) => (
              <label
                key={id}
                className="flex items-center gap-1 text-[9px] text-[var(--foreground)] cursor-pointer select-none"
              >
                <input
                  type="checkbox"
                  checked={!!vars[id]}
                  onChange={() => onToggle(id)}
                  className="accent-[var(--primary)]"
                />
                {label}
              </label>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

export function ExportDialog() {
  const { toast } = useToast();

  const imageDir = useAppStore((s) => s.imageDir);
  const numFrames = useAppStore((s) => s.numFrames);
  const exportActive = useAppStore((s) => s.exportActive);
  const exportProgress = useAppStore((s) => s.exportProgress);
  const displayComponent = useAppStore((s) => s.displayComponent);
  const hasStrain = useAppStore((s) => s.hasStrain);
  const strainComponents = useAppStore((s) => s.strainComponents);
  const vis = useAppStore((s) => s.visSettings);
  const arrows = useAppStore((s) => s.arrowSettings);

  // --- Scientific export state ---
  const [sciPath, setSciPath] = useState("");
  const sciTouched = useRef(false);
  const [sciFormat, setSciFormat] = useState<"h5" | "mat" | "npz">("h5");
  // Which variables to write into the export file. Keys match the backend's
  // ALL_EXPORTABLE_VARS. Default = all selected (matches previous behaviour).
  const [sciVars, setSciVars] = useState<Record<string, boolean>>({
    X_ref: true, Y_ref: true, ROI_mask: true,
    U: true, V: true, Magnitude: true, Velocity: true,
    exx: true, eyy: true, exy: true,
    e1: true, e2: true, max_shear: true, von_mises: true, rotation: true,
  });
  const toggleSciVar = (k: string) =>
    setSciVars((p) => ({ ...p, [k]: !p[k] }));
  const setSciVarsAll = (value: boolean) =>
    setSciVars((p) =>
      Object.fromEntries(Object.keys(p).map((k) => [k, value])) as typeof p,
    );
  const selectedSciVarList = (): string[] =>
    Object.entries(sciVars).filter(([, v]) => v).map(([k]) => k);

  // --- Image export state ---
  const [imgDir, setImgDir] = useState("");
  const imgTouched = useRef(false);
  const [frameStart, setFrameStart] = useState("1");
  const [frameEnd, setFrameEnd] = useState(String(Math.max(1, numFrames)));
  const [dpi, setDpi] = useState("150");

  // --- Component selection with per-component ranges ---
  const [selectedComponents, setSelectedComponents] = useState<
    Record<string, CompEntry>
  >({});

  // --- Animation export state ---
  const [animPath, setAnimPath] = useState("");
  const animTouched = useRef(false);
  const [animFormat, setAnimFormat] = useState<"gif" | "mp4">("gif");
  const [animComponent, setAnimComponent] = useState<string>("u");
  const [animFps, setAnimFps] = useState("10");
  const [animResize, setAnimResize] = useState("1.0");
  const [animColorbar, setAnimColorbar] = useState(true);
  const [animTimestamp, setAnimTimestamp] = useState(false);

  // --- Colorbar settings from store (shared with live display) ---
  const cbs = useAppStore((s) => s.colorbarSettings);

  // --- Track which export type is active (to show progress in correct section) ---
  const [activeExportType, setActiveExportType] = useState<"images" | "animation" | null>(null);

  // --- Report export state ---
  const [reportPath, setReportPath] = useState("");
  const reportTouched = useRef(false);
  const [reportBusy, setReportBusy] = useState(false);
  const ALL_REPORT_SECTIONS = [
    "header", "experiment", "parameters", "roi",
    "key_frames", "statistics", "probes",
    "line_probes", "area_probes",
  ] as const;
  const SECTION_LABELS: Record<string, string> = {
    header: "Header",
    experiment: "Experiment",
    parameters: "Parameters",
    roi: "ROI",
    key_frames: "Key Frames",
    statistics: "Statistics",
    probes: "Point Probes",
    line_probes: "Line Probes",
    area_probes: "Area Probes",
  };
  const [reportSections, setReportSections] = useState<Record<string, boolean>>(
    Object.fromEntries(ALL_REPORT_SECTIONS.map((s) => [s, true]))
  );
  // Custom info
  const [reportTitle, setReportTitle] = useState("RAFT-DIC Analysis Report");
  const [reportAuthor, setReportAuthor] = useState("");
  const [reportNotes, setReportNotes] = useState("");
  // Theme and format
  const [reportTheme, setReportTheme] = useState<"light" | "dark" | "academic" | "minimal">("light");
  const [reportFormat, setReportFormat] = useState<"html" | "pdf" | "both">("html");
  // Key frame components (default to magnitude only)
  const [keyFrameComponents, setKeyFrameComponents] = useState<Record<string, boolean>>({
    magnitude: true,
  });

  // --- FileBrowser state ---
  const [browseTarget, setBrowseTarget] = useState<"sci" | "img" | "anim" | "report" | null>(null);

  // Helper: generate timestamp string (YYYY-MM-DD_HHmmss)
  const makeTimestamp = () => {
    const d = new Date();
    const date = d.toISOString().slice(0, 10);
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    const ss = String(d.getSeconds()).padStart(2, "0");
    return { date, timestamp: `${date}_${hh}${mm}${ss}` };
  };

  // Default paths based on imageDir (only if user hasn't manually edited)
  useEffect(() => {
    if (imageDir) {
      const normalized = imageDir.replace(/\\/g, "/").replace(/\/+$/, "");
      const { date, timestamp } = makeTimestamp();
      if (!sciTouched.current) setSciPath(`${normalized}/results.${sciFormat}`);
      if (!imgTouched.current) setImgDir(`${normalized}/export/`);
      if (!animTouched.current) setAnimPath(`${normalized}/animation_${animComponent}_${animFps}fps_${timestamp}.${animFormat}`);
      const reportExt = reportFormat === "pdf" ? "pdf" : "html";
      if (!reportTouched.current) setReportPath(`${normalized}/report_${date}.${reportExt}`);
    }
  }, [imageDir]);

  // Update scientific export path extension when format changes
  useEffect(() => {
    if (sciTouched.current) {
      setSciPath((p) => p.replace(/\.(h5|hdf5|mat|npz)$/i, `.${sciFormat}`));
    } else if (imageDir) {
      const normalized = imageDir.replace(/\\/g, "/").replace(/\/+$/, "");
      setSciPath(`${normalized}/results.${sciFormat}`);
    }
  }, [sciFormat]);

  // Update report path extension when format changes (if not manually edited)
  useEffect(() => {
    if (!reportTouched.current && imageDir) {
      const normalized = imageDir.replace(/\\/g, "/").replace(/\/+$/, "");
      const { date } = makeTimestamp();
      const reportExt = reportFormat === "pdf" ? "pdf" : "html";
      setReportPath(`${normalized}/report_${date}.${reportExt}`);
    } else if (reportTouched.current) {
      // Update extension in existing path
      const newExt = reportFormat === "pdf" ? "pdf" : "html";
      setReportPath((p) => p.replace(/\.(html|pdf)$/i, `.${newExt}`));
    }
  }, [reportFormat]);

  // Update frameEnd when numFrames changes
  useEffect(() => {
    setFrameEnd(String(Math.max(1, numFrames)));
  }, [numFrames]);

  // Update animation filename when component/format/fps changes (if not manually edited)
  useEffect(() => {
    if (!animTouched.current && imageDir) {
      const normalized = imageDir.replace(/\\/g, "/").replace(/\/+$/, "");
      const { timestamp } = makeTimestamp();
      setAnimPath(`${normalized}/animation_${animComponent}_${animFps}fps_${timestamp}.${animFormat}`);
    }
  }, [animComponent, animFormat, animFps, imageDir]);

  // Clear active export type when export finishes
  useEffect(() => {
    if (!exportActive) setActiveExportType(null);
  }, [exportActive]);

  // Initialize selected components with current displayComponent
  useEffect(() => {
    setSelectedComponents((prev) => {
      if (Object.values(prev).some((c) => c.enabled)) return prev;
      return { [displayComponent]: { enabled: true, vmin: "", vmax: "", auto: true } };
    });
  }, [displayComponent]);

  // Available strain components (intersection of computed and known)
  const availableStrain = hasStrain
    ? STRAIN_COMPONENTS.filter((c) => strainComponents.includes(c.value))
    : [];

  const toggleComponent = (key: string) => {
    setSelectedComponents((prev) => ({
      ...prev,
      [key]: prev[key]
        ? { ...prev[key], enabled: !prev[key].enabled }
        : { enabled: true, vmin: "", vmax: "", auto: true },
    }));
  };

  const updateComponentRange = (
    key: string,
    field: "vmin" | "vmax" | "auto",
    value: string | boolean
  ) => {
    setSelectedComponents((prev) => ({
      ...prev,
      [key]: {
        ...(prev[key] || { enabled: true, vmin: "", vmax: "", auto: true }),
        [field]: value,
      },
    }));
  };

  // --- Handlers ---

  const handleScientificExport = async () => {
    if (!sciPath.trim()) return;
    const vars = selectedSciVarList();
    if (vars.length === 0) {
      toast("error", "Select at least one variable to export");
      return;
    }
    try {
      await exportScientific({
        file_path: sciPath.trim(),
        upsample_strain: true,
        variables: vars,
      });
      toast("success", "Scientific data exported successfully");
    } catch (e: any) {
      if (e?.response?.status === 409 && e?.response?.data?.exists) {
        if (window.confirm(`File already exists:\n${sciPath}\n\nOverwrite?`)) {
          try {
            await exportScientific({
              file_path: sciPath.trim(),
              upsample_strain: true,
              overwrite: true,
              variables: vars,
            });
            toast("success", "Scientific data exported (overwritten)");
          } catch (e2) {
            const msg = e2 instanceof Error ? e2.message : "Unknown error";
            toast("error", `Export failed: ${msg}`);
          }
        }
      } else {
        const msg = e?.response?.data?.error || (e instanceof Error ? e.message : "Unknown error");
        toast("error", `Scientific export failed: ${msg}`);
      }
    }
  };

  const handleImageExport = async () => {
    if (!imgDir.trim()) return;

    // Build components dict with per-component ranges
    const components: Record<string, { vmin?: number; vmax?: number }> = {};
    for (const [key, comp] of Object.entries(selectedComponents)) {
      if (!comp?.enabled) continue;
      const entry: { vmin?: number; vmax?: number } = {};
      if (!comp.auto) {
        const vmin = parseFloat(comp.vmin);
        const vmax = parseFloat(comp.vmax);
        if (!isNaN(vmin)) entry.vmin = vmin;
        if (!isNaN(vmax)) entry.vmax = vmax;
      }
      components[key] = entry;
    }

    if (Object.keys(components).length === 0) {
      toast("error", "Select at least one component to export");
      return;
    }

    // Build colorbar settings from store
    const colorbar_settings: Record<string, unknown> = {
      font_size: cbs.tickFontSize,
      label_font_size: cbs.labelFontSize,
      num_ticks: cbs.tickCount || undefined,
      bar_thickness: cbs.barThickness,
      hide_outline: cbs.hideOutline,
      discrete_levels: cbs.discreteLevels || undefined,
      label_text: cbs.labelText || undefined,
    };

    // Build WYSIWYG settings from current visualization state
    const settings: Record<string, unknown> = {
      colormap: vis.colormap,
      alpha: vis.alpha,
      display_mode: vis.background,
      use_physical_units: vis.physicalEnabled,
      physical_ratio: vis.physicalRatio,
      physical_unit: vis.physicalUnit,
      fps: vis.fps,
      use_log_scale: vis.logScale,
      show_quiver: arrows.showQuiver,
      show_streamlines: arrows.showStreamlines,
      vector_source: displayComponent === "velocity" ? "velocity" : "displacement",
      arrow_spacing: arrows.spacing,
      arrow_scale: arrows.scale,
      arrow_color: arrows.color,
      arrow_width: arrows.lineWidth,
      include_colorbar: true,
      format: "png",
      dpi: parseInt(dpi),
      colorbar_settings,
    };

    try {
      setActiveExportType("images");
      await exportImages({
        output_dir: imgDir.trim(),
        components,
        frame_range: [parseInt(frameStart), parseInt(frameEnd)],
        settings,
      });
      toast("success", "Image export started");
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      toast("error", `Image export failed: ${msg}`);
    }
  };

  const handleCancel = async () => {
    try {
      await cancelExport();
    } catch {
      // ignore — export may have already completed
    }
  };

  const handleAnimationExport = async () => {
    if (!animPath.trim()) return;

    // Build colorbar settings from store
    const colorbar_settings: Record<string, unknown> = {
      font_size: cbs.tickFontSize,
      label_font_size: cbs.labelFontSize,
      num_ticks: cbs.tickCount || undefined,
      bar_thickness: cbs.barThickness,
      hide_outline: cbs.hideOutline,
      discrete_levels: cbs.discreteLevels || undefined,
      label_text: cbs.labelText || undefined,
    };

    const settings: Record<string, unknown> = {
      colormap: vis.colormap,
      alpha: vis.alpha,
      background: vis.background,
      log_scale: vis.logScale,
      physical_ratio: vis.physicalEnabled ? vis.physicalRatio : 1.0,
      fps: vis.fps,
      colorbar_settings,
    };

    // Respect user's Fixed Range — pass vmin/vmax if set
    if (vis.fixedRange) {
      const vmin = parseFloat(vis.vminU);
      const vmax = parseFloat(vis.vmaxU);
      if (!isNaN(vmin)) settings.vmin = vmin;
      if (!isNaN(vmax)) settings.vmax = vmax;
    }

    try {
      setActiveExportType("animation");
      await exportAnimation({
        output_path: animPath.trim(),
        format: animFormat,
        component: animComponent,
        frame_range: [parseInt(frameStart) - 1, parseInt(frameEnd) - 1],
        fps: parseInt(animFps) || 10,
        include_colorbar: animColorbar,
        timestamp_overlay: animTimestamp,
        resize_factor: parseFloat(animResize) || 1.0,
        settings,
      });
      toast("success", "Animation export started");
    } catch (e: any) {
      toast("error", e?.response?.data?.error ?? "Animation export failed");
    }
  };

  const handleReportExport = async () => {
    if (!reportPath.trim()) return;
    setReportBusy(true);
    try {
      const sections = Object.entries(reportSections)
        .filter(([, v]) => v)
        .map(([k]) => k);

      // Build key frame components list
      const kfComps = Object.entries(keyFrameComponents)
        .filter(([, v]) => v)
        .map(([k]) => k);

      const result = await exportReport({
        output_path: reportPath.trim(),
        sections: sections.length < ALL_REPORT_SECTIONS.length ? sections : undefined,
        format: reportFormat,
        theme: reportTheme,
        custom_title: reportTitle || undefined,
        author: reportAuthor || undefined,
        notes: reportNotes || undefined,
        key_frame_components: kfComps.length > 0 ? kfComps : undefined,
      });

      const msg = reportFormat === "both"
        ? `Report generated: ${result.path} + PDF`
        : `Report generated: ${result.path}`;
      toast("success", msg);
    } catch (e: any) {
      toast("error", e?.response?.data?.error ?? "Report generation failed");
    } finally {
      setReportBusy(false);
    }
  };

  const allComponents = [...DISPLACEMENT_COMPONENTS, ...availableStrain];
  const animComponents = [...DISPLACEMENT_COMPONENTS, ...availableStrain];

  return (
    <CollapsibleSection title="Export" defaultOpen={false}>
      {/* --- Scientific Data Export --- */}
      <div className="space-y-1">
        <span className="text-[10px] font-medium text-[var(--muted-foreground)] uppercase tracking-wide">
          Scientific Data
        </span>

        {/* Format selector */}
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-[var(--muted-foreground)]">Format:</span>
          <select
            value={sciFormat}
            onChange={(e) => setSciFormat(e.target.value as "h5" | "mat" | "npz")}
            className="flex-1 px-1 py-0.5 rounded text-[9px] bg-[var(--input)] border border-[var(--border)] text-[var(--foreground)]"
            title="HDF5 is recommended for large sequences (chunked + streaming)"
          >
            <option value="h5">HDF5 (.h5)  — recommended for large sequences</option>
            <option value="mat">MATLAB v7.3 (.mat)  — MATLAB R2011a+</option>
            <option value="npz">NumPy (.npz)  — Python only</option>
          </select>
        </div>

        <div className="flex gap-1">
          <input
            type="text"
            value={sciPath}
            onChange={(e) => {
              sciTouched.current = true;
              setSciPath(e.target.value);
            }}
            placeholder={`Export path (.${sciFormat})`}
            className={inputClass + " flex-1"}
          />
          <button
            onClick={() => setBrowseTarget("sci")}
            className="px-1.5 shrink-0 bg-[var(--secondary)] hover:bg-[var(--secondary)]/80 rounded text-[var(--muted-foreground)]"
            title="Browse"
          >
            <FolderOpen size={12} />
          </button>
        </div>

        {/* Variables to export */}
        <ScientificVariablePicker
          vars={sciVars}
          onToggle={toggleSciVar}
          onAll={() => setSciVarsAll(true)}
          onNone={() => setSciVarsAll(false)}
          hasStrain={hasStrain}
        />

        <button
          onClick={handleScientificExport}
          disabled={!sciPath.trim() || selectedSciVarList().length === 0}
          className={btnClass}
        >
          <Download size={12} /> Export .{sciFormat} ({selectedSciVarList().length} vars)
        </button>
      </div>

      {/* --- Divider --- */}
      <div className="h-px bg-[var(--border)] my-2" />

      {/* --- Visualization Image Export --- */}
      <div className="space-y-1.5">
        <span className="text-[10px] font-medium text-[var(--muted-foreground)] uppercase tracking-wide">
          Visualization Images
        </span>

        {/* Output directory */}
        <div className="flex gap-1">
          <input
            type="text"
            value={imgDir}
            onChange={(e) => {
              imgTouched.current = true;
              setImgDir(e.target.value);
            }}
            placeholder="Output directory"
            className={inputClass + " flex-1"}
          />
          <button
            onClick={() => setBrowseTarget("img")}
            className="px-1.5 shrink-0 bg-[var(--secondary)] hover:bg-[var(--secondary)]/80 rounded text-[var(--muted-foreground)]"
            title="Browse"
          >
            <FolderOpen size={12} />
          </button>
        </div>

        {/* Frame range + DPI */}
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-[var(--muted-foreground)]">Frames:</span>
          <SmallInput value={frameStart} onChange={setFrameStart} className="w-10" />
          <span className="text-[10px] text-[var(--muted-foreground)]">—</span>
          <SmallInput value={frameEnd} onChange={setFrameEnd} className="w-10" />
          <div className="w-px h-3 bg-[var(--border)] mx-0.5" />
          <span className="text-[10px] text-[var(--muted-foreground)]">DPI:</span>
          <select
            value={dpi}
            onChange={(e) => setDpi(e.target.value)}
            className="px-1 py-0.5 rounded text-[9px] bg-[var(--input)] border border-[var(--border)] text-[var(--foreground)]"
          >
            <option value="100">100</option>
            <option value="150">150</option>
            <option value="300">300</option>
            <option value="600">600</option>
          </select>
        </div>

        {/* Component checkboxes with per-component range */}
        <div className="space-y-0.5">
          <span className="text-[10px] text-[var(--muted-foreground)]">Components:</span>
          {allComponents.map((c) => {
            const comp = selectedComponents[c.value];
            const enabled = !!comp?.enabled;
            return (
              <div key={c.value}>
                <div className="flex items-center gap-1">
                  <label className="flex items-center gap-1 text-[10px] text-[var(--foreground)] cursor-pointer flex-1">
                    <input
                      type="checkbox"
                      checked={enabled}
                      onChange={() => toggleComponent(c.value)}
                      className="accent-[var(--primary)] w-3 h-3"
                    />
                    {c.label}
                  </label>
                  {enabled && (
                    <label className="flex items-center gap-0.5 text-[9px] text-[var(--muted-foreground)] cursor-pointer">
                      <input
                        type="checkbox"
                        checked={comp?.auto ?? true}
                        onChange={() =>
                          updateComponentRange(c.value, "auto", !(comp?.auto ?? true))
                        }
                        className="accent-[var(--primary)] w-2.5 h-2.5"
                      />
                      auto
                    </label>
                  )}
                </div>
                {enabled && !(comp?.auto ?? true) && (
                  <div className="flex items-center gap-1 ml-4 mt-0.5">
                    <input
                      type="text"
                      value={comp?.vmin ?? ""}
                      onChange={(e) =>
                        updateComponentRange(c.value, "vmin", e.target.value)
                      }
                      placeholder="min"
                      className="w-14 h-5 bg-[var(--input)] border border-[#3a3d45] rounded px-1 text-[9px] text-[var(--foreground)]"
                    />
                    <span className="text-[9px] text-[var(--muted-foreground)]">~</span>
                    <input
                      type="text"
                      value={comp?.vmax ?? ""}
                      onChange={(e) =>
                        updateComponentRange(c.value, "vmax", e.target.value)
                      }
                      placeholder="max"
                      className="w-14 h-5 bg-[var(--input)] border border-[#3a3d45] rounded px-1 text-[9px] text-[var(--foreground)]"
                    />
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* WYSIWYG note */}
        <p className="text-[9px] text-[var(--muted-foreground)] italic">
          Uses current visualization settings (colormap, overlay, arrows)
        </p>

        {/* Export button / progress */}
        {exportActive && activeExportType === "images" ? (
          <div className="space-y-1">
            <div className="h-1.5 bg-[var(--input)] rounded-full overflow-hidden">
              <div
                className="h-full bg-[var(--primary)] transition-all duration-300"
                style={{ width: `${exportProgress}%` }}
              />
            </div>
            <div className="flex items-center gap-1">
              <span className="text-[10px] text-[var(--muted-foreground)] flex-1">
                Exporting... {exportProgress.toFixed(0)}%
              </span>
              <button
                onClick={handleCancel}
                className="px-2 py-1 bg-red-900/30 hover:bg-red-900/50 rounded text-[10px] text-red-400"
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <button
            onClick={handleImageExport}
            disabled={!imgDir.trim() || exportActive}
            className={btnClass}
          >
            <Download size={12} /> Export Images
          </button>
        )}
      </div>

      {/* --- Divider --- */}
      <div className="h-px bg-[var(--border)] my-2" />

      {/* --- Animation Export --- */}
      <div className="space-y-1.5">
        <span className="text-[10px] font-medium text-[var(--muted-foreground)] uppercase tracking-wide">
          Animation
        </span>

        {/* Output path */}
        <div className="flex gap-1">
          <input
            type="text"
            value={animPath}
            onChange={(e) => { animTouched.current = true; setAnimPath(e.target.value); }}
            placeholder="Output file (.gif / .mp4)"
            className={inputClass + " flex-1"}
          />
          <button
            onClick={() => setBrowseTarget("anim")}
            className="px-1.5 shrink-0 bg-[var(--secondary)] hover:bg-[var(--secondary)]/80 rounded text-[var(--muted-foreground)]"
            title="Browse"
          >
            <FolderOpen size={12} />
          </button>
        </div>

        {/* Format + Component */}
        <div className="flex items-center gap-1">
          <select
            value={animFormat}
            onChange={(e) => {
              const fmt = e.target.value as "gif" | "mp4";
              setAnimFormat(fmt);
              // Update extension in path
              setAnimPath((p) => p.replace(/\.(gif|mp4)$/i, `.${fmt}`));
            }}
            className="px-1 py-0.5 rounded text-[9px] bg-[var(--input)] border border-[var(--border)] text-[var(--foreground)]"
          >
            <option value="gif">GIF</option>
            <option value="mp4">MP4</option>
          </select>
          <select
            value={animComponent}
            onChange={(e) => setAnimComponent(e.target.value)}
            className="flex-1 px-1 py-0.5 rounded text-[9px] bg-[var(--input)] border border-[var(--border)] text-[var(--foreground)]"
          >
            {animComponents.map((c) => (
              <option key={c.value} value={c.value}>{c.label}</option>
            ))}
          </select>
        </div>

        {/* FPS + Resolution */}
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-[var(--muted-foreground)]">FPS:</span>
          <SmallInput value={animFps} onChange={setAnimFps} className="w-10" />
          <div className="w-px h-3 bg-[var(--border)] mx-0.5" />
          <span className="text-[10px] text-[var(--muted-foreground)]">Scale:</span>
          <select
            value={animResize}
            onChange={(e) => setAnimResize(e.target.value)}
            className="px-1 py-0.5 rounded text-[9px] bg-[var(--input)] border border-[var(--border)] text-[var(--foreground)]"
          >
            <option value="1.0">100%</option>
            <option value="0.5">50%</option>
            <option value="0.25">25%</option>
          </select>
        </div>

        {/* Options */}
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-1 text-[10px] text-[var(--foreground)] cursor-pointer">
            <input type="checkbox" checked={animColorbar} onChange={() => setAnimColorbar(!animColorbar)}
              className="accent-[var(--primary)] w-3 h-3" />
            Colorbar
          </label>
          <label className="flex items-center gap-1 text-[10px] text-[var(--foreground)] cursor-pointer">
            <input type="checkbox" checked={animTimestamp} onChange={() => setAnimTimestamp(!animTimestamp)}
              className="accent-[var(--primary)] w-3 h-3" />
            Timestamp
          </label>
        </div>

        {/* Export button / progress */}
        {exportActive && activeExportType === "animation" ? (
          <div className="space-y-1">
            <div className="h-1.5 bg-[var(--input)] rounded-full overflow-hidden">
              <div className="h-full bg-[var(--primary)] transition-all duration-300"
                style={{ width: `${exportProgress}%` }} />
            </div>
            <div className="flex items-center gap-1">
              <span className="text-[10px] text-[var(--muted-foreground)] flex-1">
                Exporting... {exportProgress.toFixed(0)}%
              </span>
              <button onClick={handleCancel}
                className="px-2 py-1 bg-red-900/30 hover:bg-red-900/50 rounded text-[10px] text-red-400">
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <button onClick={handleAnimationExport} disabled={!animPath.trim() || exportActive} className={btnClass}>
            <Film size={12} /> Export Animation
          </button>
        )}
      </div>

      {/* --- Divider --- */}
      <div className="h-px bg-[var(--border)] my-2" />

      {/* --- Report Generator --- */}
      <div className="space-y-1.5">
        <span className="text-[10px] font-medium text-[var(--muted-foreground)] uppercase tracking-wide">
          Report
        </span>

        {/* Custom info */}
        <div className="space-y-1">
          <input
            type="text"
            value={reportTitle}
            onChange={(e) => setReportTitle(e.target.value)}
            placeholder="RAFT-DIC Analysis Report"
            className={inputClass}
          />
          <input
            type="text"
            value={reportAuthor}
            onChange={(e) => setReportAuthor(e.target.value)}
            placeholder="Author / Institution"
            className={inputClass}
          />
          <textarea
            value={reportNotes}
            onChange={(e) => setReportNotes(e.target.value)}
            placeholder="Analysis notes..."
            rows={3}
            className={inputClass + " h-auto py-1 resize-none"}
          />
        </div>

        {/* Section checkboxes */}
        <div className="space-y-0.5">
          <span className="text-[10px] text-[var(--muted-foreground)]">Sections:</span>
          <div className="grid grid-cols-2 gap-x-2 gap-y-0.5">
            {ALL_REPORT_SECTIONS.map((sec) => (
              <label key={sec} className="flex items-center gap-1 text-[10px] text-[var(--foreground)] cursor-pointer">
                <input
                  type="checkbox"
                  checked={reportSections[sec]}
                  onChange={() => setReportSections((p) => ({ ...p, [sec]: !p[sec] }))}
                  className="accent-[var(--primary)] w-3 h-3"
                />
                {SECTION_LABELS[sec] ?? sec}
              </label>
            ))}
          </div>
        </div>

        {/* Key frame component checkboxes (shown only when key_frames is checked) */}
        {reportSections.key_frames && (
          <div className="space-y-0.5">
            <span className="text-[10px] text-[var(--muted-foreground)]">Key Frame Components:</span>
            <div className="grid grid-cols-2 gap-x-2 gap-y-0.5">
              {[...DISPLACEMENT_COMPONENTS, ...availableStrain].map((c) => (
                <label key={c.value} className="flex items-center gap-1 text-[10px] text-[var(--foreground)] cursor-pointer">
                  <input
                    type="checkbox"
                    checked={!!keyFrameComponents[c.value]}
                    onChange={() =>
                      setKeyFrameComponents((p) => ({
                        ...p,
                        [c.value]: !p[c.value],
                      }))
                    }
                    className="accent-[var(--primary)] w-3 h-3"
                  />
                  {c.label}
                </label>
              ))}
            </div>
          </div>
        )}

        {/* Output settings: theme + format */}
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-[var(--muted-foreground)]">Theme:</span>
          <select
            value={reportTheme}
            onChange={(e) => setReportTheme(e.target.value as typeof reportTheme)}
            className="flex-1 px-1 py-0.5 rounded text-[9px] bg-[var(--input)] border border-[var(--border)] text-[var(--foreground)]"
          >
            <option value="light">Light</option>
            <option value="dark">Dark</option>
            <option value="academic">Academic</option>
            <option value="minimal">Minimal</option>
          </select>
          <div className="w-px h-3 bg-[var(--border)] mx-0.5" />
          <span className="text-[10px] text-[var(--muted-foreground)]">Format:</span>
          <select
            value={reportFormat}
            onChange={(e) => setReportFormat(e.target.value as typeof reportFormat)}
            className="flex-1 px-1 py-0.5 rounded text-[9px] bg-[var(--input)] border border-[var(--border)] text-[var(--foreground)]"
          >
            <option value="html">HTML</option>
            <option value="pdf">PDF</option>
            <option value="both">Both</option>
          </select>
        </div>

        {/* Output path */}
        <div className="flex gap-1">
          <input
            type="text"
            value={reportPath}
            onChange={(e) => { reportTouched.current = true; setReportPath(e.target.value); }}
            placeholder={`Output file (.${reportFormat === "pdf" ? "pdf" : "html"})`}
            className={inputClass + " flex-1"}
          />
          <button
            onClick={() => setBrowseTarget("report")}
            className="px-1.5 shrink-0 bg-[var(--secondary)] hover:bg-[var(--secondary)]/80 rounded text-[var(--muted-foreground)]"
            title="Browse"
          >
            <FolderOpen size={12} />
          </button>
        </div>

        <button
          onClick={handleReportExport}
          disabled={!reportPath.trim() || reportBusy}
          className={btnClass}
        >
          <FileText size={12} /> {reportBusy ? "Generating..." : "Generate Report"}
        </button>
      </div>

      {/* FileBrowser modal */}
      {browseTarget && (
        <FileBrowser
          open
          onClose={() => setBrowseTarget(null)}
          onSelect={(path) => {
            if (browseTarget === "sci") { sciTouched.current = true; setSciPath(path); }
            else if (browseTarget === "img") { imgTouched.current = true; setImgDir(path); }
            else if (browseTarget === "anim") { animTouched.current = true; setAnimPath(path); }
            else if (browseTarget === "report") { reportTouched.current = true; setReportPath(path); }
            setBrowseTarget(null);
          }}
          initialPath={
            browseTarget === "sci" ? sciPath.replace(/\/[^/]*$/, "")
            : browseTarget === "anim" ? animPath.replace(/\/[^/]*$/, "")
            : browseTarget === "report" ? reportPath.replace(/\/[^/]*$/, "")
            : imgDir
          }
          {...(browseTarget === "sci" ? {
            mode: "file" as const,
            fileFilter: [".h5", ".hdf5", ".mat", ".npz"],
            defaultFileName: `results.${sciFormat}`,
          } : {})}
          {...(browseTarget === "anim" ? {
            mode: "file" as const,
            fileFilter: [".gif", ".mp4"],
            defaultFileName: `animation_${animComponent}_${animFps}fps_${makeTimestamp().timestamp}.${animFormat}`,
          } : {})}
          {...(browseTarget === "report" ? {
            mode: "file" as const,
            fileFilter: reportFormat === "pdf" ? [".pdf"] : [".html"],
            defaultFileName: `report_${makeTimestamp().date}.${reportFormat === "pdf" ? "pdf" : "html"}`,
          } : {})}
        />
      )}
    </CollapsibleSection>
  );
}
