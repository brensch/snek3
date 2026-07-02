import { useEffect, useMemo, useState } from "react";
import type { RunConfig } from "../types";
import { ConfigFields } from "./ConfigFields";

type Props = {
  config: RunConfig | null;
  onSave?: (config: RunConfig) => Promise<void>;
  readOnly?: boolean;
};

// The training knobs. Visibility is controlled by the caller (the run page's
// "Configure" button); edits are staged locally so changing a field only updates
// the draft, and nothing is sent to the backend until Save. For the live run this
// applies at the next generation boundary; historical runs are read-only.
export function ConfigPanel({ config, onSave, readOnly }: Props) {
  const [draft, setDraft] = useState<RunConfig | null>(config);
  const [busy, setBusy] = useState(false);

  const dirty = useMemo(
    () => !!draft && !!config && !readOnly && JSON.stringify(draft) !== JSON.stringify(config),
    [draft, config, readOnly],
  );

  // Sync the draft from the backend whenever there are no pending local edits, so
  // live-config polling keeps the form current without clobbering in-progress edits.
  useEffect(() => {
    if (!dirty) setDraft(config);
  }, [config, dirty]);

  if (!config || !draft) {
    return <section className="card p-4 text-sm text-ink-3">No config on disk.</section>;
  }

  const setField = (key: keyof RunConfig, value: number | boolean) =>
    setDraft((prev) => (prev ? { ...prev, [key]: value } : prev));

  async function save() {
    if (!onSave || !draft) return;
    setBusy(true);
    try {
      await onSave(draft);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="card p-4">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className="card-title">Training knobs</span>
        <span className="text-xs text-ink-3">
          {readOnly ? "read-only (run not live)" : "applied at generation boundaries"}
        </span>
        {dirty && (
          <div className="ml-auto flex items-center gap-2">
            <span className="text-xs text-warn">unsaved changes</span>
            <button className="btn" disabled={busy} onClick={() => setDraft(config)}>
              Discard
            </button>
            <button
              className="btn-primary"
              disabled={busy}
              onClick={save}
            >
              {busy ? "Saving…" : "Save changes"}
            </button>
          </div>
        )}
      </div>
      <ConfigFields config={draft} onChange={setField} disabled={busy || readOnly} />
    </section>
  );
}
