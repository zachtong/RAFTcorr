interface SliderFieldProps {
  value: number;
  onChange: (val: number) => void;
  min?: number;
  max?: number;
  step?: number;
  format?: (v: number) => string;
}

export function SliderField({
  value,
  onChange,
  min = 0,
  max = 1,
  step = 0.01,
  format = (v) => v.toFixed(2),
}: SliderFieldProps) {
  return (
    <div className="flex items-center gap-2 w-full">
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="flex-1 h-1 bg-[var(--secondary)] rounded appearance-none cursor-pointer accent-[var(--primary)]"
      />
      <span className="text-[11px] text-[var(--muted-foreground)] w-8 text-right">
        {format(value)}
      </span>
    </div>
  );
}
