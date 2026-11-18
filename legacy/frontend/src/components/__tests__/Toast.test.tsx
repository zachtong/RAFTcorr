import { render, screen, act, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ToastProvider, useToast } from "@/components/shared/Toast";

// ---- Helper component to trigger toasts from tests --------------------

function ToastTrigger({
  type,
  message,
}: {
  type: "success" | "error" | "info" | "warning";
  message: string;
}) {
  const { toast } = useToast();
  return (
    <button onClick={() => toast(type, message)}>
      Trigger {type}
    </button>
  );
}

// ---- Tests ------------------------------------------------------------

describe("ToastProvider / useToast", () => {
  it("renders children without any toasts initially", () => {
    render(
      <ToastProvider>
        <div>App Content</div>
      </ToastProvider>
    );
    expect(screen.getByText("App Content")).toBeInTheDocument();
  });

  it("shows a success toast when triggered", async () => {
    const user = userEvent.setup();
    render(
      <ToastProvider>
        <ToastTrigger type="success" message="Operation succeeded" />
      </ToastProvider>
    );

    await user.click(screen.getByText("Trigger success"));
    expect(screen.getByText("Operation succeeded")).toBeInTheDocument();
  });

  it("shows an error toast when triggered", async () => {
    const user = userEvent.setup();
    render(
      <ToastProvider>
        <ToastTrigger type="error" message="Something went wrong" />
      </ToastProvider>
    );

    await user.click(screen.getByText("Trigger error"));
    expect(screen.getByText("Something went wrong")).toBeInTheDocument();
  });

  it("auto-dismisses non-warning toast after 4000ms", () => {
    vi.useFakeTimers();
    render(
      <ToastProvider>
        <ToastTrigger type="info" message="Info message" />
      </ToastProvider>
    );

    // Use fireEvent (synchronous) to avoid timer interaction with userEvent
    fireEvent.click(screen.getByText("Trigger info"));
    expect(screen.getByText("Info message")).toBeInTheDocument();

    // Advance just before the timeout — still visible
    act(() => {
      vi.advanceTimersByTime(3999);
    });
    expect(screen.getByText("Info message")).toBeInTheDocument();

    // Advance past 4000ms — dismissed
    act(() => {
      vi.advanceTimersByTime(2);
    });
    expect(screen.queryByText("Info message")).not.toBeInTheDocument();

    vi.useRealTimers();
  });

  it("auto-dismisses warning toast after 8000ms (longer)", () => {
    vi.useFakeTimers();
    render(
      <ToastProvider>
        <ToastTrigger type="warning" message="Warning message" />
      </ToastProvider>
    );

    fireEvent.click(screen.getByText("Trigger warning"));
    expect(screen.getByText("Warning message")).toBeInTheDocument();

    // Still visible at 4000ms (non-warning would be gone)
    act(() => {
      vi.advanceTimersByTime(4000);
    });
    expect(screen.getByText("Warning message")).toBeInTheDocument();

    // Still visible just before 8000ms
    act(() => {
      vi.advanceTimersByTime(3999);
    });
    expect(screen.getByText("Warning message")).toBeInTheDocument();

    // Gone at 8000ms
    act(() => {
      vi.advanceTimersByTime(2);
    });
    expect(screen.queryByText("Warning message")).not.toBeInTheDocument();

    vi.useRealTimers();
  });

  it("dismiss button removes toast immediately", async () => {
    const user = userEvent.setup();
    render(
      <ToastProvider>
        <ToastTrigger type="success" message="Dismiss me" />
      </ToastProvider>
    );

    await user.click(screen.getByText("Trigger success"));
    expect(screen.getByText("Dismiss me")).toBeInTheDocument();

    // The X close button — find the button inside the toast
    const closeButton = screen
      .getByText("Dismiss me")
      .parentElement!.querySelector("button");
    expect(closeButton).toBeTruthy();
    await user.click(closeButton!);

    expect(screen.queryByText("Dismiss me")).not.toBeInTheDocument();
  });

  it("can show multiple toasts simultaneously", async () => {
    const user = userEvent.setup();
    render(
      <ToastProvider>
        <ToastTrigger type="success" message="Toast A" />
        <ToastTrigger type="error" message="Toast B" />
      </ToastProvider>
    );

    await user.click(screen.getByText("Trigger success"));
    await user.click(screen.getByText("Trigger error"));

    expect(screen.getByText("Toast A")).toBeInTheDocument();
    expect(screen.getByText("Toast B")).toBeInTheDocument();
  });

  it("useToast returns a no-op when used outside provider", () => {
    // Rendering a trigger outside ToastProvider — toast() should not throw
    function Standalone() {
      const { toast } = useToast();
      return <button onClick={() => toast("info", "orphan")}>Fire</button>;
    }

    render(<Standalone />);
    // Clicking should not throw (default context has a no-op toast)
    expect(() => screen.getByText("Fire")).not.toThrow();
  });
});
