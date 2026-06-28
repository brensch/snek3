import React, { useEffect, useState } from "react";
import { api } from "./api.js";

// Live control for the running trainer: status + pause/resume/stop, and an
// editable grid of the live-tunable params (applied at the next generation).
// Token is stored in localStorage and sent on write requests.
export default function ControlPanel({ status, params, liveKeys, lockedKeys, live }) {
  const [token, setToken] = useState(() => localStorage.getItem("snek_token") || "");
  const [draft, setDraft] = useState({});
  const [msg, setMsg] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => { localStorage.setItem("snek_token", token); }, [token]);

  if (!live) {
    return <p className="muted">Viewing an archived run — controls apply to the live run only.</p>;
  }

  const flash = (text, ok = true) => { setMsg({ text, ok }); setTimeout(() => setMsg(null), 4000); };

  const doControl = async (action) => {
    setBusy(true);
    try { await api.control(action, token); flash(`${action} sent`); }
    catch (e) { flash(String(e.message || e), false); }
    finally { setBusy(false); }
  };

  const applyParams = async () => {
    const patch = {};
    for (const [k, v] of Object.entries(draft)) {
      if (v !== "" && v != null) patch[k] = Number(v);
    }
    if (!Object.keys(patch).length) { flash("nothing changed", false); return; }
    setBusy(true);
    try {
      const r = await api.setParams(patch, token);
      const rej = Object.entries(r.rejected || {});
      flash(rej.length ? `applied; rejected: ${rej.map(([k, why]) => `${k} (${why})`).join(", ")}`
                       : `applied ${Object.keys(r.applied).join(", ")}`, !rej.length);
      setDraft({});
    } catch (e) { flash(String(e.message || e), false); }
    finally { setBusy(false); }
  };

  const paused = status?.paused;
  const running = status?.running;
  const prog = status?.progress;

  return (
    <div className="control">
      <div className="control-row">
        <span className={"pill " + (running ? (paused ? "warn" : "live") : "done")}>
          {running ? (paused ? "❚❚ paused" : "● running") : "■ stopped"}
          {status?.generation != null ? ` · gen ${status.generation}` : ""}
        </span>
        {status?.phase && <span className="muted">{status.phase}</span>}
        {prog && prog.total > 1 && (
          <span className="progress"><i style={{ width: `${Math.min(100, 100 * prog.done / prog.total)}%` }} /></span>
        )}
        <div className="grow" />
        {paused
          ? <button disabled={busy} onClick={() => doControl("resume")}>Resume</button>
          : <button disabled={busy} onClick={() => doControl("pause")}>Pause</button>}
        <button className="danger" disabled={busy}
          onClick={() => { if (confirm("Stop the run? It will finish the current generation and exit.")) doControl("stop"); }}>
          Stop
        </button>
      </div>

      <div className="control-row">
        <input type="password" placeholder="write token" value={token}
          onChange={(e) => setToken(e.target.value)} className="token" />
        {msg && <span className={msg.ok ? "ok" : "err"}>{msg.text}</span>}
      </div>

      <div className="params">
        {(liveKeys || []).map((k) => (
          <label key={k} className="param">
            <span>{k}</span>
            <input type="number" step="any"
              placeholder={params?.[k] ?? ""}
              value={draft[k] ?? ""}
              onChange={(e) => setDraft((d) => ({ ...d, [k]: e.target.value }))} />
          </label>
        ))}
      </div>
      <div className="control-row">
        <button disabled={busy} onClick={applyParams}>Apply params (next gen)</button>
        <span className="muted">locked: {(lockedKeys || []).join(", ")}</span>
      </div>
    </div>
  );
}
