// The dashboard's one event stream. The server multiplexes stats frames,
// trainer log lines and live league-game updates over a single SSE connection
// — one per tab, because browsers cap HTTP/1.1 connections per host (~6) and
// holding three streams per tab starved every other API call.
//
// Subscribers register per-kind handlers; the connection is opened on the
// first subscriber and closed when the last one leaves.
import { fromBinary } from "@bufbuild/protobuf";
import { StatsFrameSchema } from "../gen/snek_pb";
import type { StatsFrame } from "../gen/snek_pb";
import type { LiveEval } from "./eval";

export type LogEntry = { t_unix_ms: number; message: string };

export type EventHandlers = {
  stats?: (frame: StatsFrame) => void;
  log?: (entry: LogEntry) => void;
  eval?: (status: LiveEval) => void;
};

let source: EventSource | null = null;
const subscribers = new Set<EventHandlers>();

function dispatch<K extends keyof EventHandlers>(kind: K, value: Parameters<NonNullable<EventHandlers[K]>>[0]) {
  for (const handlers of subscribers) {
    (handlers[kind] as ((v: typeof value) => void) | undefined)?.(value);
  }
}

function ensureOpen() {
  if (source) return;
  source = new EventSource("/api/stream/events");
  source.addEventListener("stats", (event) => {
    try {
      const raw = atob((event as MessageEvent).data);
      const bytes = Uint8Array.from(raw, (c) => c.charCodeAt(0));
      dispatch("stats", fromBinary(StatsFrameSchema, bytes));
    } catch {
      /* ignore malformed frames */
    }
  });
  source.addEventListener("log", (event) => {
    try {
      dispatch("log", JSON.parse((event as MessageEvent).data) as LogEntry);
    } catch {
      /* ignore malformed lines */
    }
  });
  source.addEventListener("eval", (event) => {
    try {
      dispatch("eval", JSON.parse((event as MessageEvent).data) as LiveEval);
    } catch {
      /* ignore malformed frames */
    }
  });
}

/** Subscribe to the shared event stream; returns an unsubscribe function. */
export function subscribeEvents(handlers: EventHandlers): () => void {
  subscribers.add(handlers);
  ensureOpen();
  return () => {
    subscribers.delete(handlers);
    if (subscribers.size === 0) {
      source?.close();
      source = null;
    }
  };
}
