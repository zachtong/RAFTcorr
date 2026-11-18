interface ProgressOverlayProps {
  active: boolean;
  percent: number;
  label?: string;
}

export function ProgressOverlay({ active, percent, label }: ProgressOverlayProps) {
  if (!active) return null;

  return (
    <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="bg-[var(--card)] rounded-lg p-6 min-w-[280px] flex flex-col items-center gap-3">
        <div className="text-sm font-medium">{label ?? "Processing..."}</div>
        <div className="w-full h-2 bg-[var(--muted)] rounded-full overflow-hidden">
          <div
            className="h-full bg-[var(--primary)] rounded-full transition-all duration-200"
            style={{ width: `${percent}%` }}
          />
        </div>
        <div className="text-xs text-[var(--muted-foreground)]">
          {percent.toFixed(1)}%
        </div>
      </div>
    </div>
  );
}
