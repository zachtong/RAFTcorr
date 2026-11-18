import { type ReactNode } from "react";
import { SessionControls } from "@/components/shared/SessionControls";

interface SidebarProps {
  children: ReactNode;
  width?: number;
}

export function Sidebar({ children, width = 260 }: SidebarProps) {
  return (
    <aside
      className="flex-shrink-0 border-r border-[var(--border)] bg-[var(--card)] flex flex-col"
      style={{ width }}
    >
      <SessionControls />
      {children}
    </aside>
  );
}
