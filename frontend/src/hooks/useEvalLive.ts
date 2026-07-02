import { useEffect, useState } from "react";
import { openEvalStream } from "../api/eval";
import type { LiveEval } from "../api/eval";

// Subscribe to the evaluation league's live-match stream while `enabled`
// (i.e. when viewing the trainer's active run — the league only plays there).
export function useEvalLive(enabled: boolean) {
  const [status, setStatus] = useState<LiveEval | null>(null);
  useEffect(() => {
    if (!enabled) {
      setStatus(null);
      return;
    }
    const stream = openEvalStream(setStatus);
    return () => stream.close();
  }, [enabled]);
  return status;
}
