import React, { useState } from "react";
import { api } from "./api.js";

// Live control for the running trainer: status + pause/resume/stop, and an
// editable grid of the live-tunable params (applied at the next generation).
export default function ControlPanel({ run, status, params, liveKeys, lockedKeys, live }) {
  const [draft, setDraft] = useState({});
  const [msg, setMsg] = useState(null);
  const [busy, setBusy] = useState(false);

  const flash = (text, ok = true) => { setMsg({ text, ok }); setTimeout(() => setMsg(null), 4000); };

  // Archived run: not the live one. Offer to resume it from its checkpoint
  // (continues weights + gen count; switches the trainer to it).
  if (!live) {
    const resume = async () => {
      setBusy(true);
      try { await api.resumeRun(run); flash(`resuming ${run}…`); }
      catch (e) { flash(String(e.message || e), false); }
      finally { setBusy(false); }
    };
    return (
      <div className="control">
        <p className="muted">This is an archived run — it isn't currently training.</p>
        <div className="control-row">
          <button disabled={busy} onClick={resume}>Resume this run</button>
          {msg && <span className={msg.ok ? "ok" : "err"}>{msg.text}</span>}
        </div>
        <p className="muted">Resuming continues from the saved checkpoint and makes it the live run (any currently-live run is checkpointed first).</p>
      </div>
    );
  }

  const doControl = async (action) => {
    setBusy(true);
    try { await api.control(action); flash(`${action} sent`); }
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
      const r = await api.setParams(patch);
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
  const stateClass = running ? (paused ? "paused" : "running") : "stopped";
  const stateWord = running ? (paused ? "PAUSED" : "RUNNING") : "STOPPED";
  const determinate = prog && prog.total > 1;
  const pct = determinate ? Math.min(100, Math.round(100 * prog.done / prog.total)) : null;
  const busyPhase = prog && prog.total <= 1;  // training/eval/recording = indeterminate

  const last = status?.last || {};
  const chip = (label, key, fmt = (v) => v) =>
    last[key] != null ? <span className="chip" key={key}><em>{label}</em><b>{fmt(last[key])}</b></span> : null;

  return (
    <div className="control">
      <div className={"hero " + stateClass}>
        <div className="hero-main">
          <span className={"state-dot " + stateClass} />
          <span className="state-word">{stateWord}</span>
          {status?.generation != null && <span className="state-gen">gen {status.generation}</span>}
          <div className="grow" />
          {running && (paused
            ? <button disabled={busy} onClick={() => doControl("resume")}>Resume</button>
            : <button disabled={busy} onClick={() => doControl("pause")}>Pause</button>)}
          {running && <button className="danger" disabled={busy}
            onClick={() => { if (confirm("Stop the run? It will finish the current generation and exit.")) doControl("stop"); }}>
            Stop
          </button>}
        </div>

        <div className="hero-phase">
          <span className="phase-now">{status?.phase || "—"}</span>
          {determinate && <span className="phase-count">{prog.done.toLocaleString()} / {prog.total.toLocaleString()}</span>}
        </div>
        <div className={"hero-bar " + (busyPhase ? "busy" : "")}>
          <i style={determinate ? { width: `${pct}%` } : undefined} />
          {determinate && <span className="bar-pct">{pct}%</span>}
        </div>

        <div className="hero-stats">
          {chip("game len", "proxy_game_len")}
          {chip("games", "proxy_games")}
          {chip("draw %", "proxy_draw_rate", (v) => `${Math.round(v * 100)}%`)}
          {chip("entropy", "target_entropy")}
          {chip("gen s", "gen_seconds")}
          {chip("GPU %", "selfplay_gpu_pct", (v) => `${v}%`)}
          {chip("resp v base", "response_vs_baseline")}
          {chip("resp v uct", "response_vs_uct")}
          {chip("proxy v base", "proxy_vs_baseline")}
          {chip("proxy v uct", "proxy_vs_uct")}
        </div>
      </div>

      {msg && <div className="control-row"><span className={msg.ok ? "ok" : "err"}>{msg.text}</span></div>}

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
