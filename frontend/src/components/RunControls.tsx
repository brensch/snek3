import { useState } from "react";

type Props = {
  running: boolean;
  stopping: boolean;
  onResume: () => Promise<void>;
  onStop: () => Promise<void>;
};

// Run-scoped training controls. A run is either the trainer's live run (which
// can be stopped) or a stored run that can be resumed to make it live again.
// While a stop is draining, the button is disabled until the loop has fully
// stopped. The "live" chip lives on the generation-progress panel.
export function RunControls({ running, stopping, onResume, onStop }: Props) {
  const [busy, setBusy] = useState(false);

  async function invoke(action: () => Promise<void>) {
    setBusy(true);
    try {
      await action();
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex items-center gap-2">
      {running ? (
        <button className="btn-danger" disabled={busy || stopping} onClick={() => invoke(onStop)}>
          {stopping ? "Stopping…" : "Stop training"}
        </button>
      ) : (
        <button className="btn" disabled={busy} onClick={() => invoke(onResume)}>
          Resume training
        </button>
      )}
    </div>
  );
}
