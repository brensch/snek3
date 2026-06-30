import type { GameSnapshot, GamesSnapshot, Phase, Point, Snake, StatsFrame } from "../types";

const phases: Phase[] = ["idle", "playing", "training", "checkpoint", "stopping", "stopped"];

class Reader {
  private offset = 0;
  constructor(private readonly bytes: Uint8Array) {}
  done() { return this.offset >= this.bytes.length; }
  varint(): bigint {
    let out = 0n;
    let shift = 0n;
    while (!this.done()) {
      const b = BigInt(this.bytes[this.offset++]);
      out |= (b & 0x7fn) << shift;
      if ((b & 0x80n) === 0n) break;
      shift += 7n;
    }
    return out;
  }
  double(): number {
    const v = new DataView(this.bytes.buffer, this.bytes.byteOffset + this.offset, 8).getFloat64(0, true);
    this.offset += 8;
    return v;
  }
  message(): Reader {
    const len = Number(this.varint());
    const start = this.offset;
    this.offset += len;
    return new Reader(this.bytes.slice(start, start + len));
  }
  skip(wire: number) {
    if (wire === 0) this.varint();
    else if (wire === 1) this.offset += 8;
    else if (wire === 2) this.message();
    else if (wire === 5) this.offset += 4;
  }
}

export function decodeStatsFrame(data: string): StatsFrame {
  const r = new Reader(fromBase64(data));
  const out: StatsFrame = emptyStats();
  while (!r.done()) {
    const tag = Number(r.varint());
    const f = tag >> 3, w = tag & 7;
    if (f === 1) out.t_unix_ms = Number(r.varint());
    else if (f === 2) out.generation = Number(r.varint());
    else if (f === 3) out.phase = phases[Number(r.varint())] ?? "idle";
    else if (f === 4) out.inferences_per_sec = r.double();
    else if (f === 5) out.games_per_sec = r.double();
    else if (f === 6) out.completed_games_total = Number(r.varint());
    else if (f === 7) out.samples_collected = Number(r.varint());
    else if (f === 8) out.samples_target = Number(r.varint());
    else if (f === 9) out.gpu_busy_pct = r.double();
    else if (f === 10) out.batch_avg_rows = Number(r.varint());
    else if (f === 11) out.policy_loss = r.double();
    else if (f === 12) out.value_loss = r.double();
    else if (f === 13) out.target_entropy = r.double();
    else if (f === 14) out.gpu_rows_per_sec = r.double();
    else r.skip(w);
  }
  return out;
}

export function decodeGamesSnapshot(data: string): GamesSnapshot {
  const r = new Reader(fromBase64(data));
  const out: GamesSnapshot = { t_unix_ms: 0, games: [] };
  while (!r.done()) {
    const tag = Number(r.varint());
    const f = tag >> 3, w = tag & 7;
    if (f === 1) out.t_unix_ms = Number(r.varint());
    else if (f === 2) out.games.push(readGame(r.message()));
    else r.skip(w);
  }
  return out;
}

function readGame(r: Reader): GameSnapshot {
  const g: GameSnapshot = { id: 0, turn: 0, board_w: 0, board_h: 0, snakes: [], food: [] };
  while (!r.done()) {
    const tag = Number(r.varint());
    const f = tag >> 3, w = tag & 7;
    if (f === 1) g.id = Number(r.varint());
    else if (f === 2) g.turn = Number(r.varint());
    else if (f === 3) g.board_w = Number(r.varint());
    else if (f === 4) g.board_h = Number(r.varint());
    else if (f === 5) g.snakes.push(readSnake(r.message()));
    else if (f === 6) g.food.push(readPoint(r.message()));
    else r.skip(w);
  }
  return g;
}

function readSnake(r: Reader): Snake {
  const s: Snake = { alive: false, health: 0, body: [] };
  while (!r.done()) {
    const tag = Number(r.varint());
    const f = tag >> 3, w = tag & 7;
    if (f === 1) s.alive = r.varint() !== 0n;
    else if (f === 2) s.health = Number(r.varint());
    else if (f === 3) s.body.push(readPoint(r.message()));
    else r.skip(w);
  }
  return s;
}

function readPoint(r: Reader): Point {
  const p: Point = { x: 0, y: 0 };
  while (!r.done()) {
    const tag = Number(r.varint());
    const f = tag >> 3, w = tag & 7;
    if (f === 1) p.x = Number(r.varint());
    else if (f === 2) p.y = Number(r.varint());
    else r.skip(w);
  }
  return p;
}

function fromBase64(data: string): Uint8Array {
  const raw = atob(data);
  return Uint8Array.from(raw, (c) => c.charCodeAt(0));
}

function emptyStats(): StatsFrame {
  return { t_unix_ms: 0, generation: 0, phase: "idle", inferences_per_sec: 0, games_per_sec: 0, completed_games_total: 0, samples_collected: 0, samples_target: 0, gpu_busy_pct: 0, batch_avg_rows: 0, policy_loss: 0, value_loss: 0, target_entropy: 0, gpu_rows_per_sec: 0 };
}
