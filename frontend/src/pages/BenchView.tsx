import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { openBenchStream } from "../api/bench";
import type { BenchEvent } from "../api/bench";
import { number, rate } from "../lib/format";

type Row = { batch: number; rowsPerSec: number; callsPerSec: number; meanMs: number };
type Status = "connecting" | "running" | "done" | "error";

// /bench — runs a GPU batch-size throughput sweep and shows inf/s per batch size,
// filling the table live and highlighting the fastest once the sweep finishes.
export function BenchView() {
  const [status, setStatus] = useState<Status>("connecting");
  const [error, setError] = useState<string | null>(null);
  const [device, setDevice] = useState<string>("");
  const [trunk, setTrunk] = useState<{ channels: number; blocks: number } | null>(null);
  const [seconds, setSeconds] = useState(0);
  const [batches, setBatches] = useState<number[]>([]);
  const [rows, setRows] = useState<Row[]>([]);
  const [measuring, setMeasuring] = useState<{ batch: number; index: number; total: number } | null>(null);
  const [best, setBest] = useState<number | null>(null);
  const streamRef = useRef<EventSource | null>(null);

  const onEvent = useCallback((ev: BenchEvent) => {
    switch (ev.kind) {
      case "start":
        setStatus("running");
        setBatches(ev.batches);
        setSeconds(ev.seconds);
        setDevice(ev.device);
        setTrunk({ channels: ev.trunk_channels, blocks: ev.trunk_blocks });
        break;
      case "measuring":
        setMeasuring({ batch: ev.batch, index: ev.index, total: ev.total });
        break;
      case "result":
        setRows((prev) => [
          ...prev,
          { batch: ev.batch, rowsPerSec: ev.rows_per_sec, callsPerSec: ev.calls_per_sec, meanMs: ev.mean_ms },
        ]);
        break;
      case "done":
        setBest(ev.best_batch);
        setMeasuring(null);
        setStatus("done");
        streamRef.current?.close();
        break;
      case "error":
        setError(ev.detail);
        setStatus("error");
        streamRef.current?.close();
        break;
    }
  }, []);

  const start = useCallback(() => {
    streamRef.current?.close();
    setStatus("connecting");
    setError(null);
    setBatches([]);
    setRows([]);
    setMeasuring(null);
    setBest(null);
    setDevice("");
    setTrunk(null);
    streamRef.current = openBenchStream(onEvent);
  }, [onEvent]);

  useEffect(() => {
    start();
    return () => streamRef.current?.close();
    // Run once on mount; `start` is stable via useCallback.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const inProgress = status === "connecting" || status === "running";
  const done = rows.length;
  const total = batches.length || measuring?.total || 0;

  return (
    <div className="mx-auto max-w-3xl px-5 py-6">
      <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-lg font-semibold text-ink">GPU batch-size benchmark</h1>
          <p className="text-sm text-ink-3">
            Times a forward pass (H2D → net → D2H) at each batch size on the current GPU to find the
            highest inference throughput.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Link to="/" className="btn">
            Back to runs
          </Link>
          <button className="btn" disabled={inProgress} onClick={start}>
            {inProgress ? "Running…" : "Run again"}
          </button>
        </div>
      </header>

      {(device || trunk) && (
        <div className="mb-4 flex flex-wrap gap-x-6 gap-y-1 text-xs text-ink-3">
          {device && <span>device {device}</span>}
          {trunk && <span>trunk {trunk.channels}×{trunk.blocks}</span>}
          {seconds > 0 && <span>{seconds}s per batch</span>}
        </div>
      )}

      {status === "error" && (
        <div className="mb-4 card border-bad/40 p-3 text-sm text-bad">
          {error ?? "Benchmark failed."}
        </div>
      )}

      {inProgress && (
        <div className="mb-4">
          <div className="mb-1 flex items-center justify-between text-xs text-ink-3">
            <span>
              {measuring
                ? `Measuring batch ${number(measuring.batch)}…`
                : "Warming up the GPU…"}
            </span>
            <span className="font-mono">
              {done}/{total || "?"}
            </span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded bg-inset">
            <div
              className="h-full bg-accent transition-all"
              style={{ width: total ? `${(done / total) * 100}%` : "8%" }}
            />
          </div>
        </div>
      )}

      {(rows.length > 0 || inProgress) && (
        <div className="card overflow-x-auto">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="border-b border-white/10 bg-inset text-left text-[10px] uppercase tracking-wide text-ink-3">
                <th className="px-3 py-2 text-right font-medium">Batch size</th>
                <th className="px-3 py-2 text-right font-medium">Inf/s (rows)</th>
                <th className="px-3 py-2 text-right font-medium">Forwards/s</th>
                <th className="px-3 py-2 text-right font-medium">Mean ms</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const isBest = status === "done" && best === row.batch;
                return (
                  <tr
                    key={row.batch}
                    className={`border-b border-white/5 last:border-0 ${
                      isBest ? "bg-good/10 text-good" : "text-ink-2"
                    }`}
                  >
                    <td className="px-3 py-2 text-right font-mono">
                      {number(row.batch)}
                      {isBest && (
                        <span className="ml-2 rounded-full bg-good/15 px-2 py-0.5 text-[10px] font-medium text-good">
                          best
                        </span>
                      )}
                    </td>
                    <td className="px-3 py-2 text-right font-mono">{number(row.rowsPerSec)}</td>
                    <td className="px-3 py-2 text-right font-mono">{rate(row.callsPerSec)}</td>
                    <td className="px-3 py-2 text-right font-mono">{row.meanMs.toFixed(3)}</td>
                  </tr>
                );
              })}
              {inProgress && measuring && !rows.some((r) => r.batch === measuring.batch) && (
                <tr className="border-b border-white/5 text-ink-3 last:border-0">
                  <td className="px-3 py-2 text-right font-mono">{number(measuring.batch)}</td>
                  <td className="px-3 py-2 text-right font-mono" colSpan={3}>
                    measuring…
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
