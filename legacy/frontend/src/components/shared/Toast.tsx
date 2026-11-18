import { useEffect, useState, useCallback, createContext, useContext, type ReactNode } from "react";
import { X } from "lucide-react";

type ToastType = "success" | "error" | "info" | "warning";

interface ToastItem {
  id: number;
  type: ToastType;
  message: string;
}

interface ToastContextValue {
  toast: (type: ToastType, message: string) => void;
}

const ToastContext = createContext<ToastContextValue>({ toast: () => {} });

export function useToast() {
  return useContext(ToastContext);
}

let nextId = 0;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const toast = useCallback((type: ToastType, message: string) => {
    const id = nextId++;
    setToasts((prev) => [...prev, { id, type, message }]);
  }, []);

  const dismiss = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}
      <div className="fixed bottom-4 right-4 z-[100] flex flex-col gap-2 max-w-sm">
        {toasts.map((t) => (
          <ToastItem key={t.id} item={t} onDismiss={dismiss} />
        ))}
      </div>
    </ToastContext.Provider>
  );
}

function ToastItem({ item, onDismiss }: { item: ToastItem; onDismiss: (id: number) => void }) {
  useEffect(() => {
    const duration = item.type === "warning" ? 8000 : 4000;
    const timer = setTimeout(() => onDismiss(item.id), duration);
    return () => clearTimeout(timer);
  }, [item.id, item.type, onDismiss]);

  const colors = {
    success: "border-green-500/40 bg-green-500/10 text-green-300",
    error: "border-red-500/40 bg-red-500/10 text-red-300",
    info: "border-blue-500/40 bg-blue-500/10 text-blue-300",
    warning: "border-yellow-500/40 bg-yellow-500/10 text-yellow-300",
  };

  return (
    <div
      className={`flex items-start gap-2 px-3 py-2 rounded-lg border backdrop-blur-sm animate-in slide-in-from-right ${colors[item.type]}`}
    >
      <span className="text-[12px] flex-1">{item.message}</span>
      <button
        onClick={() => onDismiss(item.id)}
        className="opacity-60 hover:opacity-100 shrink-0"
      >
        <X size={12} />
      </button>
    </div>
  );
}
