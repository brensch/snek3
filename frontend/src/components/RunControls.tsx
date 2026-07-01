import { useState } from "react";

type Props = {
  live: boolean;
  running: boolean;
  stopping: boolean;
  onResume: () => Promise<void>;
  onStop: () => Promise<void>;
};

// Run-scoped training controls. A run is either the trainer's live run (which
// can be stopped) or a stored run that can be resumed to make it live again.
// While a stop is draining, the button is disabled until the loop has fully
// stopped.
export function RunControls({ live, running, stopping, onResume, onStop }: Props) {
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
      {live && running && !stopping ? (
        <span className="rounded-full bg-green-500/15 px-2 py-0.5 text-xs font-medium text-green-400">● live</span>
      ) : null}
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
