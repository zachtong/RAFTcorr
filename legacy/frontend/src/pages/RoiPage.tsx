import { Sidebar } from "@/components/layout/Sidebar";
import { PathSettings } from "@/components/roi/PathSettings";
import { ProcessingParams } from "@/components/roi/ProcessingParams";
import { RunControl } from "@/components/roi/RunControl";
import { RoiCanvas } from "@/components/roi/RoiCanvas";
import { RoiToolbar } from "@/components/roi/RoiToolbar";
import { RoiImportDialog } from "@/components/roi/RoiImportDialog";
import { BatchRoiImportDialog } from "@/components/roi/BatchRoiImportDialog";
import { FrameRoiSelector } from "@/components/roi/FrameRoiSelector";
import { OnboardingOverlay } from "@/components/shared/OnboardingOverlay";

export function RoiPage() {
  return (
    <div className="flex flex-1 overflow-hidden">
      <Sidebar>
        <div className="flex flex-col flex-1 overflow-y-auto">
          <PathSettings />
          <FrameRoiSelector />
          <ProcessingParams />
        </div>
        <RunControl />
      </Sidebar>
      <div className="flex-1 flex flex-col">
        <RoiCanvas />
        <RoiToolbar />
      </div>
      <RoiImportDialog />
      <BatchRoiImportDialog />
      <OnboardingOverlay />
    </div>
  );
}
