// GPU batch-size benchmark SSE. The trainer streams progress as JSON events
// (same contract as the log stream): a `start`, one `measuring`/`result` pair per
// batch size, then `done` — or a single `error` if a run is active. Opening the
// stream starts the sweep; closing the EventSource lets it finish server-side.
export type BenchEvent =
  | {
      kind: "start";
      batches: number[];
      seconds: number;
      device: string;
      trunk_channels: number;
      trunk_blocks: number;
    }
  | { kind: "measuring"; batch: number; index: number; total: number }
  | {
      kind: "result";
      batch: number;
      rows_per_sec: number;
      calls_per_sec: number;
      mean_ms: number;
    }
  | { kind: "done"; best_batch: number }
  | { kind: "error"; detail: string };

export function openBenchStream(onEvent: (event: BenchEvent) => void): EventSource {
  const events = new EventSource("/api/bench/stream");
  events.addEventListener("bench", (event) => {
    try {
      onEvent(JSON.parse((event as MessageEvent).data) as BenchEvent);
    } catch {
      /* ignore malformed lines */
    }
  });
  return events;
}
