import { lazy, Suspense } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AppLayout } from "@/components/layout/AppLayout";
import { ToastProvider } from "@/components/shared/Toast";
import { ShortcutHelpOverlay } from "@/components/shared/ShortcutHelpOverlay";
import { useSocket } from "@/hooks/useSocket";
import { useGlobalShortcuts } from "@/hooks/useGlobalShortcuts";

// Lazy load pages for code splitting (Recharts is heavy)
const RoiPage = lazy(() =>
  import("@/pages/RoiPage").then((m) => ({ default: m.RoiPage }))
);
const DisplacementPage = lazy(() =>
  import("@/pages/DisplacementPage").then((m) => ({ default: m.DisplacementPage }))
);
const PostProcessingPage = lazy(() =>
  import("@/pages/PostProcessingPage").then((m) => ({
    default: m.PostProcessingPage,
  }))
);

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, refetchOnWindowFocus: false },
  },
});

function PageFallback() {
  return (
    <div className="flex-1 flex items-center justify-center text-[var(--muted-foreground)] text-sm">
      Loading...
    </div>
  );
}

function AppInner() {
  useSocket();
  const { helpOpen, setHelpOpen } = useGlobalShortcuts();
  return (
    <>
      <Routes>
        <Route element={<AppLayout />}>
          <Route
            path="/"
            element={
              <Suspense fallback={<PageFallback />}>
                <RoiPage />
              </Suspense>
            }
          />
          <Route
            path="/displacement"
            element={
              <Suspense fallback={<PageFallback />}>
                <DisplacementPage />
              </Suspense>
            }
          />
          <Route
            path="/post-processing"
            element={
              <Suspense fallback={<PageFallback />}>
                <PostProcessingPage />
              </Suspense>
            }
          />
        </Route>
      </Routes>
      <ShortcutHelpOverlay open={helpOpen} onClose={() => setHelpOpen(false)} />
    </>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ToastProvider>
        <BrowserRouter>
          <AppInner />
        </BrowserRouter>
      </ToastProvider>
    </QueryClientProvider>
  );
}
