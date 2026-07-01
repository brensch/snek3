import { useEffect, useState } from "react";
import { getRuns } from "../api/proto";
import type { RunSummary } from "../gen/viewer_pb";

// Polls the run list so the home page reflects the live run's progress.
export function useRunList() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const reply = await getRuns();
        if (alive) {
          setRuns(reply.runs);
          setError(null);
        }
      } catch (err) {
        if (alive) setError(String(err));
      } finally {
        if (alive) setLoading(false);
      }
    };
    void load();
    const timer = window.setInterval(load, 5000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, []);

  return { runs, error, loading };
}
