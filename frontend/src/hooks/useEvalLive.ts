import { useEffect, useState } from "react";
import { subscribeEvents } from "../api/events";
import type { LiveEval } from "../api/eval";

// Live league-game updates from the shared event stream (one per board turn),
// while `enabled` (i.e. viewing the trainer's active run).
export function useEvalLive(enabled: boolean) {
  const [status, setStatus] = useState<LiveEval | null>(null);
  useEffect(() => {
    if (!enabled) {
      setStatus(null);
      return;
    }
    return subscribeEvents({ eval: setStatus });
  }, [enabled]);
  return status;
}
