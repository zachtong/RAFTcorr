interface SegmentedControlProps {
  options: string[];
  value: string;
  onChange: (val: string) => void;
}

export function SegmentedControl({ options, value, onChange }: SegmentedControlProps) {
  return (
    <div className="flex bg-[var(--input)] rounded overflow-hidden border border-[#3a3d45]">
      {options.map((opt) => (
        <button
          key={opt}
          onClick={() => onChange(opt)}
          className={`flex-1 px-2 py-1 text-[11px] transition-colors ${
            value === opt
              ? "bg-[var(--primary)] text-white"
              : "text-[var(--muted-foreground)] hover:bg-[var(--secondary)]"
          }`}
        >
          {opt}
        </button>
      ))}
    </div>
  );
}
