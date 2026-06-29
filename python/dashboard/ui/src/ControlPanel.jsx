import React, { useState } from "react";
import { api } from "./api.js";
import ParamRows from "./ParamRows.jsx";

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
        <p className="muted">This run isn't currently training.</p>
        <div className="control-row">
          <button disabled={busy} onClick={resume}>Start this run</button>
          {msg && <span className={msg.ok ? "ok" : "err"}>{msg.text}</span>}
        </div>
        <p className="muted">Starting continues from the saved checkpoint and makes it the live run.</p>
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

  const running = status?.running;
  const prog = status?.progress;
  const stopping = status?.phase === "stopping" || status?.phase === "switching";
  const stateClass = running ? "running" : "stopped";
  const stateWord = running ? "RUNNING" : "STOPPED";
  const determinate = prog && prog.total > 1;
  const pct = determinate ? Math.min(100, Math.round(100 * prog.done / prog.total)) : null;
  const busyPhase = prog && prog.total <= 1;  // training/eval/recording = indeterminate

  const last = status?.last || {};
  const chip = (label, key, fmt = (v) => v) =>
    last[key] != null ? <span className="chip" key={key}><em>{label}</em><b>{fmt(last[key])}</b></span> : null;
  const fmtNum = (v, digits = 0) => Number(v || 0).toLocaleString(undefined, {
    maximumFractionDigits: digits,
  });
  const hasInflight = prog && (
    prog.inflight_slots > 0 || prog.inflight_steps > 0 || prog.inflight_samples > 0 ||
    prog.resumed_completed_samples > 0
  );

  return (
    <div className="control">
      <div className={"hero " + stateClass}>
        <div className="hero-main">
          <span className={"state-dot " + stateClass} />
          <span className="state-word">{stateWord}</span>
          {status?.generation != null && <span className="state-gen">gen {status.generation}</span>}
          <div className="grow" />
          {running && <button className="danger" disabled={busy || stopping}
            onClick={() => doControl("stop")}>
            {stopping ? "Stopping" : "Stop"}
          </button>}
        </div>

        <div className="hero-phase">
          <span className="phase-now">{status?.phase || "—"}</span>
          {determinate && <span className="phase-count">{prog.done.toLocaleString()} / {prog.total.toLocaleString()}</span>}
        </div>
        {hasInflight && (
          <div className="hero-phase">
            <span className="phase-now">resumed in-flight</span>
            <span className="phase-count">
              {prog.resumed_completed_samples > 0 && `${fmtNum(prog.resumed_completed_samples)} completed samples · `}
              {fmtNum(prog.inflight_slots)} slots · {fmtNum(prog.inflight_steps)} steps · {fmtNum(prog.inflight_samples)} pending samples
              {prog.inflight_turn_max != null && ` · turns avg ${fmtNum(prog.inflight_turn_mean, 1)} max ${fmtNum(prog.inflight_turn_max)}`}
            </span>
          </div>
        )}
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

      <ParamRows
        keys={liveKeys || []}
        values={params || {}}
        placeholders={params || {}}
        draft={draft}
        onDraft={(k, v) => setDraft((d) => ({ ...d, [k]: v }))}
        title="Live training levers"
      />
      <div className="control-row">
        <button disabled={busy} onClick={applyParams}>Apply params (next gen)</button>
        <span className="muted">locked: {(lockedKeys || []).join(", ")}</span>
      </div>
    </div>
  );
}
