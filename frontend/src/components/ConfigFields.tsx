import { configFields } from "../lib/configFields";
import type { RunConfig } from "../types";

type Props = {
  config: RunConfig;
  onChange: (key: keyof RunConfig, value: number | boolean) => void;
  disabled?: boolean;
};

// The grid of training-knob inputs. Shared by the run page's ConfigPanel (editing
// a live/historical run) and the home page's "start fresh run" form.
export function ConfigFields({ config, onChange, disabled }: Props) {
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-5">
      {configFields.map((field) => (
        <label key={field.key} className="grid gap-1">
          <span className="flex items-baseline justify-between gap-2 text-xs text-slate-500">
            <span>{field.label}</span>
            {field.hint ? <em className="truncate text-[10px] not-italic text-slate-600">{field.hint}</em> : null}
          </span>
          {field.kind === "bool" ? (
            <input
              type="checkbox"
              checked={Boolean(config[field.key])}
              onChange={(e) => onChange(field.key, e.target.checked)}
              disabled={disabled}
            />
          ) : (
            <input
              className="input"
              type="number"
              value={String(config[field.key])}
              onChange={(e) => onChange(field.key, Number(e.target.value))}
              disabled={disabled}
            />
          )}
        </label>
      ))}
    </div>
  );
}
