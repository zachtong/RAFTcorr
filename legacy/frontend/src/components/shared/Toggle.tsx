interface ToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
}

export function Toggle({ checked, onChange }: ToggleProps) {
  return (
    <button
      onClick={() => onChange(!checked)}
      className={`w-8 h-4 rounded-full transition-colors relative ${
        checked ? "bg-[var(--primary)]" : "bg-[var(--secondary)]"
      }`}
    >
      <div
        className={`w-3 h-3 rounded-full bg-white absolute top-0.5 transition-transform ${
          checked ? "translate-x-[18px]" : "translate-x-0.5"
        }`}
      />
    </button>
  );
}
