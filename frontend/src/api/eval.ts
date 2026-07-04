// Live evaluation-league payloads (delivered over the shared event stream in
// ./events — one message per board turn while league games are in flight).
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

// One player of an in-flight game (seat s plays player s % N).
export type LiveEvalPlayer = { gen: number; name: string; elo: number; games: number };

// One in-flight game slot: its own players, current turn, latest frame.
export type LiveEvalGame = {
  seq: number;
  turn: number;
  players: LiveEvalPlayer[];
  frame: LiveFrame | null;
};

export type LiveEval = {
  /** The league is enabled and slots are running (false while paused). */
  active: boolean;
  games: LiveEvalGame[];
  updated_unix_ms: number;
};
