// Live evaluation-league SSE: the trainer publishes the in-flight arena game
// (its players with fitted Elos, current board frames) about once a second.
export type LiveFrameSnake = {
  alive: boolean;
  body: [number, number][]; // head first
  health: number;
  chosen_move: number;
  policy: number[];
  play_policy: number[];
  value: number;
};

// One board frame, same shape as a frame of the recorded games files.
export type LiveFrame = {
  turn: number;
  width: number;
  height: number;
  food: [number, number][];
  hazards: [number, number][];
  snakes: LiveFrameSnake[];
};

export type LiveEvalGame = { index: number; turn: number; frame: LiveFrame | null };

// One player of the in-flight game (seat s plays player s % N).
export type LiveEvalPlayer = { gen: number; elo: number; games: number };

export type LiveEval = {
  active: boolean;
  seq: number;
  players: LiveEvalPlayer[];
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
