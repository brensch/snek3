import React, { useEffect, useState } from "react";
import { api } from "./api.js";

// Landing page: start a new named run, or pick an existing one to view.
const NEW_FIELDS = ["board", "filters", "blocks", "depth", "count", "samples"];

export default function Home({ runs, liveRun, onSelect }) {
  const [name, setName] = useState("");
  const [draft, setDraft] = useState({});
  const [base, setBase] = useState({});
  const [msg, setMsg] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => { api.state().then((s) => s && setBase(s.base_spec || {})); }, []);

  const flash = (text, ok = true) => { setMsg({ text, ok }); setTimeout(() => setMsg(null), 5000); };

  const start = async () => {
    const nm = name.trim();
    if (!nm) { flash("name required", false); return; }
    const params = {};
    for (const [k, v] of Object.entries(draft)) if (v !== "" && v != null) params[k] = Number(v);
    setBusy(true);
    try {
      await api.createRun(nm, params);
      flash(`starting ${nm}…`);
      onSelect(nm);
    } catch (e) { flash(String(e.message || e), false); }
    finally { setBusy(false); }
  };

  const resume = async (r, e) => {
    e.stopPropagation();
    setBusy(true);
    try { await api.resumeRun(r); flash(`resuming ${r}…`); onSelect(r); }
    catch (err) { flash(String(err.message || err), false); }
    finally { setBusy(false); }
  };

  return (
    <div className="home">
      <section className="card">
        <h2>Start a new run</h2>
        <div className="control-row">
          <input className="text-input" placeholder="run name" value={name}
            onChange={(e) => setName(e.target.value)} style={{ minWidth: 240 }} />
          <button disabled={busy} onClick={start}>Start run</button>
          {msg && <span className={msg.ok ? "ok" : "err"}>{msg.text}</span>}
        </div>
        <div className="params" style={{ marginTop: 4 }}>
          {NEW_FIELDS.map((k) => (
            <label key={k} className="param">
              <span>{k}</span>
              <input type="number" step="any" placeholder={base[k] ?? ""}
                value={draft[k] ?? ""}
                onChange={(e) => setDraft((d) => ({ ...d, [k]: e.target.value }))} />
            </label>
          ))}
        </div>
        <p className="muted" style={{ marginTop: 8 }}>
          A new run starts fresh and runs until you stop it. Starting one switches
          the trainer to it (the current run, if any, is checkpointed first). Leave
          fields blank to use defaults.
        </p>
      </section>

      <section className="card">
        <h2>Runs</h2>
        {runs.length ? (
          <ul className="run-list">
            {runs.map((r) => (
              <li key={r} className="run-row">
                <button className="run-item" onClick={() => onSelect(r)}>
                  <span className={"run-dot " + (r === liveRun ? "live" : "")} />
                  <span className="run-name">{r}</span>
                  {r === liveRun && <span className="run-badge">live</span>}
                </button>
                {r !== liveRun && (
                  <button className="run-resume" disabled={busy}
                    title="Continue this run from its checkpoint"
                    onClick={(e) => resume(r, e)}>Resume</button>
                )}
              </li>
            ))}
          </ul>
        ) : <p className="muted">No runs yet — start one above.</p>}
      </section>
    </div>
  );
}
