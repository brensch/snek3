import { decodeGamesSnapshot, decodeStatsFrame } from "./protobuf";
import type { GamesSnapshot, StatsFrame } from "../types";

export function openStatsStream(onFrame: (frame: StatsFrame) => void): EventSource {
  const events = new EventSource("/api/stream/stats");
  events.addEventListener("stats", (event) => onFrame(decodeStatsFrame((event as MessageEvent).data)));
  return events;
}

export function openGamesStream(onFrame: (frame: GamesSnapshot) => void): EventSource {
  const events = new EventSource("/api/stream/games");
  events.addEventListener("games", (event) => onFrame(decodeGamesSnapshot((event as MessageEvent).data)));
  return events;
}
