interface SelectFieldProps {
  value: string;
  options: { value: string; label: string }[];
  onChange: (val: string) => void;
  className?: string;
  disabled?: boolean;
  title?: string;
}

export function SelectField({
  value,
  options,
  onChange,
  className = "",
  disabled = false,
  title,
}: SelectFieldProps) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      disabled={disabled}
      title={title}
      className={`h-6 bg-[var(--input)] border border-[#3a3d45] rounded px-1.5 text-[11px] text-[var(--foreground)] focus:border-[var(--primary)] focus:outline-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed ${className}`}
    >
      {options.map((opt) => (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  );
}
