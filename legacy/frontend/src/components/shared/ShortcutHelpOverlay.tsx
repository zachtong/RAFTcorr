import { useEffect } from "react";
import { X } from "lucide-react";

interface Shortcut {
  keys: string[];
  description: string;
}

const SHORTCUTS: { group: string; items: Shortcut[] }[] = [
  {
    group: "Playback (Displacement / Post-Processing)",
    items: [
      { keys: ["Space"], description: "Play / pause" },
      { keys: ["←", "→"], description: "Previous / next frame" },
      { keys: ["Shift", "←/→"], description: "Jump 10 frames" },
    ],
  },
  {
    group: "ROI page",
    items: [
      { keys: ["Ctrl", "Z"], description: "Undo last ROI change" },
      { keys: ["Ctrl", "Shift", "Z"], description: "Redo (Ctrl+Y also works)" },
      { keys: ["Esc"], description: "Cancel active drawing" },
    ],
  },
  {
    group: "Post-Processing page",
    items: [
      { keys: ["Esc"], description: "Cancel active probe placement" },
    ],
  },
  {
    group: "Navigation",
    items: [
      { keys: ["1"], description: "Go to ROI page" },
      { keys: ["2"], description: "Go to Displacement page (once ROI confirmed)" },
      { keys: ["3"], description: "Go to Post-Processing (once results exist)" },
      { keys: ["?"], description: "Toggle this help overlay" },
    ],
  },
];

interface ShortcutHelpOverlayProps {
  open: boolean;
  onClose: () => void;
}

export function ShortcutHelpOverlay({ open, onClose }: ShortcutHelpOverlayProps) {
  // Allow Esc to close the overlay itself.
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="w-[min(560px,92vw)] max-h-[82vh] overflow-y-auto rounded-lg border border-[var(--border)] bg-[var(--card)] shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)]">
          <h2 className="text-[13px] font-semibold text-[var(--foreground)]">
            Keyboard shortcuts
          </h2>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-[var(--secondary)] text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
            aria-label="Close"
          >
            <X size={16} />
          </button>
        </header>

        <div className="p-4 flex flex-col gap-4">
          {SHORTCUTS.map((group) => (
            <section key={group.group}>
              <h3 className="text-[10px] uppercase tracking-wider text-[var(--muted-foreground)] mb-2">
                {group.group}
              </h3>
              <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1.5 text-[11px]">
                {group.items.map((sc) => (
                  <div key={sc.description} className="contents">
                    <dt className="flex items-center gap-1">
                      {sc.keys.map((k, i) => (
                        <span key={`${k}-${i}`} className="flex items-center gap-1">
                          <kbd className="px-1.5 py-0.5 rounded border border-[var(--border)] bg-[var(--input)] text-[var(--foreground)] font-mono text-[10px] shadow-sm">
                            {k}
                          </kbd>
                          {i < sc.keys.length - 1 && (
                            <span className="text-[var(--muted-foreground)]">
                              +
                            </span>
                          )}
                        </span>
                      ))}
                    </dt>
                    <dd className="text-[var(--foreground)]">
                      {sc.description}
                    </dd>
                  </div>
                ))}
              </dl>
            </section>
          ))}
        </div>

        <footer className="px-4 py-2 border-t border-[var(--border)] text-[10px] text-[var(--muted-foreground)]">
          Typing in an input field disables these shortcuts.
        </footer>
      </div>
    </div>
  );
}
