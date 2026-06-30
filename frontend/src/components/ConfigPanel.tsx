import { useState } from "react";
import { configFields } from "../lib/configFields";
import type { RunConfig } from "../types";

type Props = { config: RunConfig | null; onSave: (config: RunConfig) => Promise<void> };

export function ConfigPanel({ config, onSave }: Props) {
  const [busy, setBusy] = useState(false);
  if (!config) return <section className="panel">Loading config...</section>;

  async function save(next: RunConfig) {
    setBusy(true);
    try { await onSave(next); } finally { setBusy(false); }
  }

  return (
    <section className="panel">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="section-title">Training knobs</h2>
        <span className="text-xs text-slate-500">applied at generation boundaries</span>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-5">
        {configFields.map((field) => (
          <label key={field.key} className="grid gap-1">
            <span className="flex items-baseline justify-between gap-2 text-xs text-slate-500">
              <span>{field.label}</span>
              {field.hint ? <em className="truncate text-[10px] not-italic text-slate-600">{field.hint}</em> : null}
            </span>
            {field.kind === "bool" ? (
              <input type="checkbox" checked={Boolean(config[field.key])} onChange={(e) => save({ ...config, [field.key]: e.target.checked })} disabled={busy} />
            ) : (
              <input className="input" type="number" value={String(config[field.key])} onChange={(e) => save({ ...config, [field.key]: Number(e.target.value) })} disabled={busy} />
            )}
          </label>
        ))}
      </div>
    </section>
  );
}
