import React, { useMemo, useState } from "react";
import Board, { snakeColor } from "./Board.jsx";
import { MOVE_ARROW, MOVE_LABEL } from "./api.js";

// Interactive walk of a replayed search tree. At each node we show, per snake,
// the decoupled-PUCT decomposition the engine used to choose where to descend:
//   Q(a) + c_puct · P(a) · sqrt(ΣN) / (1 + N(a))
// so you can see exactly why exploration went where it did. Children are the
// joint actions actually expanded; click one to descend.
export default function TreeExplorer({ data, cPuct, food, hazards, width, height }) {
  const tree = data?.tree;
  const byId = useMemo(() => {
    const m = new Map();
    tree?.nodes?.forEach((n) => m.set(n.id, n));
    return m;
  }, [tree]);
  const [path, setPath] = useState([0]); // node ids from root to current

  if (!tree) return null;
  const curId = path[path.length - 1];
  const node = byId.get(curId);
  if (!node) return null;

  const descend = (childId) => setPath((p) => [...p, childId]);
  const jumpTo = (idx) => setPath((p) => p.slice(0, idx + 1));

  // Principal variation from the *root*: repeatedly follow the heaviest child.
  const pv = useMemo(() => buildPV(byId), [byId]);

  return (
    <div className="tree">
      <div className="tree-head">
        <div className="crumbs">
          {path.map((id, idx) => {
            const n = byId.get(id);
            return (
              <React.Fragment key={idx}>
                {idx > 0 && <span className="crumb-sep">›</span>}
                <button
                  className={"crumb" + (idx === path.length - 1 ? " active" : "")}
                  onClick={() => jumpTo(idx)}
                >
                  {idx === 0 ? "root" : `d${n?.depth} #${id}`}
                </button>
              </React.Fragment>
            );
          })}
        </div>
        <div className="tree-meta">
          <Stat label="nodes" value={tree.node_count} />
          <Stat label="max depth" value={tree.max_depth} />
          <Stat label="node N" value={Math.round(node.total_visits)} />
          {node.terminal && (
            <span className="badge term">
              terminal [{(node.term_value || []).map((v) => v.toFixed(1)).join(", ")}]
            </span>
          )}
        </div>
      </div>

      <div className="tree-body">
        <div className="tree-board">
          <Board
            width={width}
            height={height}
            snakes={node.snakes}
            food={food}
            hazards={hazards}
            youIndex={data.you}
            cell={Math.max(14, Math.floor(300 / Math.max(width, height)))}
          />
          <div className="muted small">position at this node (food shown from turn start)</div>
        </div>

        <div className="tree-detail">
          {node.actions.map((rows, si) => (
            <SnakeActions key={si} si={si} rows={rows} cPuct={cPuct} you={data.you} />
          ))}

          <div className="children">
            <div className="children-h">
              explored children ({node.children.length})
            </div>
            {node.children.length === 0 && (
              <div className="muted small">leaf — not expanded further</div>
            )}
            {[...node.children]
              .map((c) => ({ ...c, n: byId.get(c.child)?.total_visits || 0 }))
              .sort((a, b) => b.n - a.n)
              .map((c) => (
                <button
                  key={c.child}
                  className={"child" + (pv.has(c.child) ? " pv" : "")}
                  onClick={() => descend(c.child)}
                >
                  <span className="child-moves">
                    {c.moves.map((m, si) => (
                      <span
                        key={si}
                        className="cmove"
                        style={{ color: snakeColor(si) }}
                        title={`snake ${si}: ${MOVE_LABEL[m]}`}
                      >
                        {MOVE_ARROW[m]}
                      </span>
                    ))}
                  </span>
                  <span className="child-n">N={Math.round(c.n)}</span>
                  {pv.has(c.child) && <span className="badge pvb">PV</span>}
                  <span className="child-go">→</span>
                </button>
              ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// One snake's action breakdown at the current node, including the PUCT score the
// engine maximizes. The highlighted row is what selection would pick next.
function SnakeActions({ si, rows, cPuct, you }) {
  if (!rows || rows.length === 0) {
    return (
      <div className="sactions">
        <div className="sactions-h">
          <span className="dot" style={{ background: snakeColor(si) }} />
          snake {si} {si === you && <span className="badge you">YOU</span>}
          <span className="muted small"> — unexpanded</span>
        </div>
      </div>
    );
  }
  // Tree action rows are objects { move, prior, visits, q }.
  const totalN = rows.reduce((a, r) => a + (r.visits || 0), 0);
  const sqrtN = Math.sqrt(Math.max(1, totalN));
  const hasPrior = rows.some((r) => r.prior > 1e-8);
  const scored = rows.map(({ move, prior, visits, q }) => {
    const masked = hasPrior && prior <= 1e-8;
    const u = cPuct * prior * (sqrtN / (1 + (visits || 0)));
    return { move, prior, visits, q, u, score: q + u, masked };
  });
  const selMove = scored
    .filter((r) => !r.masked)
    .reduce((b, r) => (r.score > (b?.score ?? -Infinity) ? r : b), null)?.move;
  const maxScore = Math.max(...scored.map((r) => (r.masked ? -Infinity : r.score)));
  const minScore = Math.min(...scored.map((r) => (r.masked ? Infinity : r.score)));

  return (
    <div className="sactions">
      <div className="sactions-h">
        <span className="dot" style={{ background: snakeColor(si) }} />
        snake {si} {si === you && <span className="badge you">YOU</span>}
        <span className="muted small"> ΣN={Math.round(totalN)}</span>
      </div>
      <table className="moves tree-moves">
        <thead>
          <tr>
            <th></th>
            <th>P</th>
            <th>N</th>
            <th>Q</th>
            <th>U</th>
            <th>Q+U</th>
          </tr>
        </thead>
        <tbody>
          {scored.map((r) => (
            <tr
              key={r.move}
              className={
                (r.masked ? "masked " : "") + (r.move === selMove ? "sel" : "")
              }
            >
              <td className="mv">
                <span className="arrow">{MOVE_ARROW[r.move]}</span> {MOVE_LABEL[r.move]}
                {r.move === selMove && <span className="play">argmax</span>}
              </td>
              <td className="num">{pct(r.prior)}</td>
              <td className="num">{r.visits ? Math.round(r.visits) : "·"}</td>
              <td className={"num " + qc(r.q)}>{sgn(r.q)}</td>
              <td className="num u">{r.u.toFixed(2)}</td>
              <td className="num score">
                <span
                  className="scorebar"
                  style={{ width: `${scoreWidth(r.score, minScore, maxScore, r.masked)}%` }}
                />
                {r.masked ? "—" : sgn(r.score)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function buildPV(byId) {
  const set = new Set();
  let cur = byId.get(0);
  let guard = 0;
  while (cur && cur.children.length && guard++ < 200) {
    let best = null;
    for (const c of cur.children) {
      const n = byId.get(c.child)?.total_visits || 0;
      if (!best || n > best.n) best = { child: c.child, n };
    }
    if (!best) break;
    set.add(best.child);
    cur = byId.get(best.child);
  }
  return set;
}

const Stat = ({ label, value }) => (
  <span className="tstat">
    <span className="tstat-v">{value}</span>
    <span className="tstat-l">{label}</span>
  </span>
);

const pct = (x) => (x == null ? "·" : `${(x * 100).toFixed(0)}%`);
const sgn = (x) => (x == null ? "·" : (x >= 0 ? "+" : "") + x.toFixed(2));
const qc = (q) => (q == null ? "" : q > 0.05 ? "pos" : q < -0.05 ? "neg" : "neu");
function scoreWidth(s, min, max, masked) {
  if (masked || !isFinite(min) || !isFinite(max) || max === min) return 0;
  return Math.round(((s - min) / (max - min)) * 100);
}
