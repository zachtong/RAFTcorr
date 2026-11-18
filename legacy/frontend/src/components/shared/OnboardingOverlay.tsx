import { useState, useCallback } from "react";

const STORAGE_KEY = "dic_onboarding_done";

interface Step {
  title: string;
  description: string;
  icon: string;
}

const STEPS: Step[] = [
  {
    title: "1. Load Images",
    description:
      "Click 'Browse' in the sidebar to select a folder containing your DIC image sequence. " +
      "Images will be sorted automatically by frame number.",
    icon: "\uD83D\uDCC2", // folder emoji
  },
  {
    title: "2. Draw ROI",
    description:
      "Use the toolbar below the canvas to draw a Region of Interest (ROI). " +
      "You can use rectangle, circle, or polygon tools. " +
      "Use the cut tool to exclude areas like grips or fixtures.",
    icon: "\u2B1C", // square emoji
  },
  {
    title: "3. Run DIC Processing",
    description:
      "Click 'Run' to start Digital Image Correlation. " +
      "The RAFT neural network will compute displacement fields for each frame. " +
      "Progress is shown in real-time via the sidebar.",
    icon: "\u25B6\uFE0F", // play emoji
  },
  {
    title: "4. Analyze Results",
    description:
      "Navigate to the Displacement and Post-Processing pages to view results. " +
      "You can compute strain fields, add measurement probes, and export data.",
    icon: "\uD83D\uDCC8", // chart emoji
  },
];

/**
 * First-visit onboarding overlay.
 *
 * Shows a 4-step workflow guide on first launch.
 * Completion is persisted to localStorage so it only appears once.
 */
export function OnboardingOverlay() {
  const [currentStep, setCurrentStep] = useState(0);
  const [dismissed, setDismissed] = useState(() => {
    try {
      return localStorage.getItem(STORAGE_KEY) === "true";
    } catch {
      return false;
    }
  });

  const dismiss = useCallback(() => {
    setDismissed(true);
    try {
      localStorage.setItem(STORAGE_KEY, "true");
    } catch {
      // localStorage unavailable — ignore
    }
  }, []);

  const next = useCallback(() => {
    if (currentStep < STEPS.length - 1) {
      setCurrentStep((s) => s + 1);
    } else {
      dismiss();
    }
  }, [currentStep, dismiss]);

  const prev = useCallback(() => {
    setCurrentStep((s) => Math.max(0, s - 1));
  }, []);

  if (dismissed) return null;

  const step = STEPS[currentStep];
  const isLast = currentStep === STEPS.length - 1;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-[420px] max-w-[90vw] bg-[var(--card)] border border-[var(--border)] rounded-xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="px-6 pt-6 pb-2">
          <h2 className="text-lg font-semibold text-[var(--foreground)]">
            Welcome to RAFTcorr
          </h2>
          <p className="text-[12px] text-[var(--muted-foreground)] mt-1">
            Digital Image Correlation powered by RAFT optical flow
          </p>
        </div>

        {/* Step content */}
        <div className="px-6 py-4">
          <div className="flex items-start gap-3">
            <span className="text-2xl mt-0.5">{step.icon}</span>
            <div>
              <h3 className="text-[13px] font-medium text-[var(--foreground)]">
                {step.title}
              </h3>
              <p className="text-[11px] text-[var(--muted-foreground)] mt-1 leading-relaxed">
                {step.description}
              </p>
            </div>
          </div>
        </div>

        {/* Step indicators */}
        <div className="flex justify-center gap-1.5 pb-3">
          {STEPS.map((_, i) => (
            <button
              key={i}
              onClick={() => setCurrentStep(i)}
              className={`w-2 h-2 rounded-full transition-colors ${
                i === currentStep
                  ? "bg-[var(--primary)]"
                  : "bg-[var(--muted-foreground)]/30 hover:bg-[var(--muted-foreground)]/50"
              }`}
            />
          ))}
        </div>

        {/* Actions */}
        <div className="flex items-center justify-between px-6 py-3 border-t border-[var(--border)] bg-[var(--card)]">
          <button
            onClick={dismiss}
            className="text-[11px] text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors"
          >
            Skip
          </button>
          <div className="flex items-center gap-2">
            {currentStep > 0 && (
              <button
                onClick={prev}
                className="px-3 py-1.5 text-[11px] text-[var(--foreground)] bg-[var(--secondary)] hover:bg-[var(--secondary)]/80 rounded transition-colors"
              >
                Back
              </button>
            )}
            <button
              onClick={next}
              className="px-4 py-1.5 text-[11px] text-white bg-[var(--primary)] hover:bg-[var(--primary)]/90 rounded transition-colors"
            >
              {isLast ? "Get Started" : "Next"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Reset onboarding state — call this to show the overlay again
 * (e.g., from a help menu).
 */
export function resetOnboarding(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}
