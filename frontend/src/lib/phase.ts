import { Phase } from "../gen/snek_pb";

const LABELS: Record<Phase, string> = {
  [Phase.IDLE]: "idle",
  [Phase.PLAYING]: "playing",
  [Phase.TRAINING]: "training",
  [Phase.CHECKPOINT]: "checkpoint",
  [Phase.STOPPING]: "stopping",
  [Phase.STOPPED]: "stopped",
};

export function phaseLabel(phase: Phase | undefined | null): string {
  return phase == null ? "-" : LABELS[phase] ?? "-";
}
