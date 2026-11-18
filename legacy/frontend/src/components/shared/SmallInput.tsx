interface SmallInputProps {
  value: string | number;
  onChange?: (val: string) => void;
  onBlur?: () => void;
  className?: string;
  placeholder?: string;
}

export function SmallInput({ value, onChange, onBlur, className = "w-14", placeholder }: SmallInputProps) {
  return (
    <input
      type="text"
      value={value}
      placeholder={placeholder}
      onChange={(e) => onChange?.(e.target.value)}
      onBlur={onBlur}
      className={`${className} h-6 bg-[var(--input)] border border-[#3a3d45] rounded px-1.5 text-[11px] text-[var(--foreground)] text-center focus:border-[var(--primary)] focus:outline-none`}
    />
  );
}
