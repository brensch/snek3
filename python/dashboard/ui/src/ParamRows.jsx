import React from "react";
import { paramInfo } from "./paramInfo.js";

export default function ParamRows({
  keys,
  values = {},
  placeholders = {},
  draft = {},
  onDraft,
  title = "Parameters",
  defaultOpen = false,
}) {
  return (
    <details className="param-panel" open={defaultOpen}>
      <summary>
        <span>{title}</span>
        <em>{(keys || []).length} knobs</em>
      </summary>
      <div className="param-rows">
        {(keys || []).map((key) => {
          const info = paramInfo(key);
          const current = values?.[key] ?? placeholders?.[key];
          return (
            <label key={key} className="param-row">
              <div className="param-copy">
                <div className="param-title">
                  <b>{key}</b>
                  <span>{info.name}</span>
                </div>
                <p>{info.summary}</p>
                <p className="muted">{info.details}</p>
                {info.faster && <p className="param-faster">{info.faster}</p>}
              </div>
              <div className="param-edit">
                <span>{current != null && current !== "" ? `current ${current}` : "blank = default"}</span>
                <input
                  type="number"
                  step="any"
                  placeholder={placeholders?.[key] ?? values?.[key] ?? ""}
                  value={draft[key] ?? ""}
                  onChange={(e) => onDraft?.(key, e.target.value)}
                />
              </div>
            </label>
          );
        })}
      </div>
    </details>
  );
}
