import { decodeStatsFrame } from "./protobuf";
import type { StatsFrame } from "../types";

export function openStatsStream(onFrame: (frame: StatsFrame) => void): EventSource {
  const events = new EventSource("/api/stream/stats");
  events.addEventListener("stats", (event) => onFrame(decodeStatsFrame((event as MessageEvent).data)));
  return events;
}
