// Live evaluation-league SSE: the trainer publishes the in-flight arena match
// (who is playing whom, which turn each game is up to) about once a second.
export type LiveEvalGame = { index: number; turn: number };

export type LiveEval = {
  active: boolean;
  seq: number;
  gen_a: number;
  gen_b: number;
  games_total: number;
  // Cumulative tally for the in-flight match; wins count for gen_a.
  wins: number;
  losses: number;
  draws: number;
  games: LiveEvalGame[];
  updated_unix_ms: number;
};

export function openEvalStream(onStatus: (status: LiveEval) => void): EventSource {
  const events = new EventSource("/api/stream/eval");
  events.addEventListener("eval", (event) => {
    try {
      onStatus(JSON.parse((event as MessageEvent).data) as LiveEval);
    } catch {
      /* ignore malformed frames */
    }
  });
  return events;
}
