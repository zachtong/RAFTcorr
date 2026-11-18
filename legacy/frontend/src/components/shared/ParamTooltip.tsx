import { useState, useRef, useCallback } from "react";
import { PARAM_DESCRIPTIONS } from "@/lib/paramDescriptions";

interface ParamTooltipProps {
  /** Parameter key — must match a key in PARAM_DESCRIPTIONS */
  paramKey: string;
  /** Optional override description (uses PARAM_DESCRIPTIONS[paramKey] if omitted) */
  description?: string;
  children?: React.ReactNode;
}

/**
 * Hoverable info icon that shows a tooltip with parameter description.
 *
 * Usage:
 *   <ParamTooltip paramKey="vsgSize" />
 *   <ParamTooltip paramKey="custom" description="My custom text" />
 */
export function ParamTooltip({ paramKey, description, children }: ParamTooltipProps) {
  const text = description ?? PARAM_DESCRIPTIONS[paramKey];
  const [visible, setVisible] = useState(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout>>(undefined);

  const show = useCallback(() => {
    clearTimeout(timeoutRef.current);
    setVisible(true);
  }, []);

  const hide = useCallback(() => {
    timeoutRef.current = setTimeout(() => setVisible(false), 150);
  }, []);

  if (!text) return null;

  return (
    <span
      className="relative inline-flex items-center"
      onMouseEnter={show}
      onMouseLeave={hide}
    >
      {children ?? (
        <span className="text-[10px] text-[var(--muted-foreground)] opacity-50 hover:opacity-100 cursor-help select-none">
          &#9432;
        </span>
      )}
      {visible && (
        <div
          className="absolute z-50 left-full ml-2 top-1/2 -translate-y-1/2 w-56 px-3 py-2 rounded-md
                     bg-[var(--card)] border border-[var(--border)] shadow-lg
                     text-[10px] text-[var(--foreground)] leading-relaxed whitespace-pre-line"
          onMouseEnter={show}
          onMouseLeave={hide}
        >
          {text}
        </div>
      )}
    </span>
  );
}
