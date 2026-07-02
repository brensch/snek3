// Live evaluation-league payloads (delivered over the shared event stream in
// ./events — one message per board turn while a league game is in flight).
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
