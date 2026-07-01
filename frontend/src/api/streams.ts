import { fromBinary } from "@bufbuild/protobuf";
import { StatsFrameSchema } from "../gen/snek_pb";
import type { StatsFrame } from "../gen/snek_pb";

// The trainer's stats SSE sends each StatsFrame as base64-encoded protobuf; we
// decode it with the buf-generated schema (same contract as the rest of the API).
export function openStatsStream(onFrame: (frame: StatsFrame) => void): EventSource {
  const events = new EventSource("/api/stream/stats");
  events.addEventListener("stats", (event) => {
    const raw = atob((event as MessageEvent).data);
    const bytes = Uint8Array.from(raw, (c) => c.charCodeAt(0));
    onFrame(fromBinary(StatsFrameSchema, bytes));
  });
  return events;
}
