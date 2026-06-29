import React from "react";
import { MOVE_LABEL, MOVE_ARROW } from "./api.js";
import { snakeColor } from "./Board.jsx";

// Per-snake search readout for the current turn: the value head, the visit-count
// policy over Up/Down/Left/Right, and (for our snake) the move actually played.
// `root_actions[snake]` rows are [move, prior, visits, q]; `root_policy` is the
// flattened [snake*4 + move] visit policy; `root_values[snake]` is the value.
export default function MovePanel({ move, roster, nSnakes }) {
  if (!move) return null;
  const s = move.search || {};
  const policy = s.root_policy || [];
  const values = s.root_values || [];
  const actions = s.root_actions || [];

  return (
    <div className="panel">
      <div className="panel-h">Search readout · turn {move.turn}</div>
      <div className="snake-cards">
        {Array.from({ length: nSnakes }).map((_, i) => {
          const name = roster?.[i]?.name || `snake ${i}`;
          const isYou = i === move.you;
          const val = values[i] ?? 0;
          const pol = policy.slice(i * 4, i * 4 + 4);
          const byMove = {};
          (actions[i] || []).forEach(([m, prior, visits, q]) => {
            byMove[m] = { prior, visits, q };
          });
          const maxVisit = Math.max(1, ...[0, 1, 2, 3].map((m) => byMove[m]?.visits || 0));
          const bestMove = [0, 1, 2, 3].reduce(
            (b, m) => ((byMove[m]?.visits || 0) > (byMove[b]?.visits || -1) ? m : b),
            0
          );
          return (
            <div className="snake-card" key={i}>
              <div className="snake-card-h">
                <span className="dot" style={{ background: snakeColor(i) }} />
                <span className="snake-name">{name}</span>
                {isYou && <span className="badge you">YOU</span>}
                <span className="spacer" />
                <ValueGauge v={val} />
              </div>
              <table className="moves">
                <thead>
                  <tr>
                    <th></th>
                    <th>policy</th>
                    <th>prior</th>
                    <th>N</th>
                    <th>Q</th>
                  </tr>
                </thead>
                <tbody>
                  {[0, 1, 2, 3].map((m) => {
                    const a = byMove[m] || {};
                    const chosen = isYou && move.chosen_move === m;
                    const best = m === bestMove && (a.visits || 0) > 0;
                    return (
                      <tr key={m} className={chosen ? "chosen" : best ? "best" : ""}>
                        <td className="mv">
                          <span className="arrow">{MOVE_ARROW[m]}</span> {MOVE_LABEL[m]}
                          {chosen && <span className="play">played</span>}
                        </td>
                        <td className="barcell">
                          <span
                            className="bar"
                            style={{ width: `${Math.round((pol[m] || 0) * 100)}%` }}
                          />
                          <span className="num">{fmtPct(pol[m])}</span>
                        </td>
                        <td className="num">{fmtPct(a.prior)}</td>
                        <td className="num">
                          <span
                            className="bar bar-n"
                            style={{ width: `${Math.round(((a.visits || 0) / maxVisit) * 100)}%` }}
                          />
                          {a.visits ? Math.round(a.visits) : "·"}
                        </td>
                        <td className={"num q " + qClass(a.q)}>{a.q != null ? fmtSigned(a.q) : "·"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ValueGauge({ v }) {
  // v in [-1, 1]; center at 0.
  const pct = Math.max(0, Math.min(100, (v + 1) * 50));
  return (
    <span className="vgauge" title={`value ${v.toFixed(3)}`}>
      <span className="vgauge-track">
        <span className="vgauge-zero" />
        <span
          className="vgauge-fill"
          style={{
            left: v >= 0 ? "50%" : `${pct}%`,
            width: `${Math.abs(pct - 50)}%`,
            background: v >= 0 ? "#22c55e" : "#ef4444",
          }}
        />
      </span>
      <span className={"vgauge-num " + qClass(v)}>{fmtSigned(v)}</span>
    </span>
  );
}

const fmtPct = (x) => (x == null ? "·" : `${(x * 100).toFixed(0)}%`);
const fmtSigned = (x) => (x >= 0 ? "+" : "") + x.toFixed(2);
const qClass = (q) => (q == null ? "" : q > 0.05 ? "pos" : q < -0.05 ? "neg" : "neu");
