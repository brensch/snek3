import { useEffect, useState } from "react";
import { getRunDetail } from "../api/proto";
import type { RunDetail } from "../gen/viewer_pb";

// Loads a run's detail (summary + metrics history + recorded game gens) and
// re-polls so a live run's charts and new game gens keep updating.
export function useRunDetail(runId: string) {
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setDetail(null);
    const load = async () => {
      try {
        const next = await getRunDetail(runId);
        if (alive) {
          setDetail(next);
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
  }, [runId]);

  return { detail, error, loading };
}
