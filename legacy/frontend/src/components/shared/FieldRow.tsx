import type { ReactNode } from "react";

interface FieldRowProps {
  label: ReactNode;
  children: ReactNode;
}

export function FieldRow({ label, children }: FieldRowProps) {
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-[11px] text-[var(--muted-foreground)] whitespace-nowrap shrink-0">
        {label}
      </span>
      <div className="flex items-center gap-1 min-w-0 flex-1 justify-end">
        {children}
      </div>
    </div>
  );
}
