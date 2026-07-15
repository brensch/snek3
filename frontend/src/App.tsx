import { Navigate, Route, Routes } from "react-router-dom";
import { BenchView } from "./pages/BenchView";
import { GameView } from "./pages/GameView";
import { RunView } from "./pages/RunView";
import { RunsHome } from "./pages/RunsHome";

export default function App() {
  return (
    <div className="min-h-screen">
      <Routes>
        <Route path="/" element={<RunsHome />} />
        <Route path="/bench" element={<BenchView />} />
        <Route path="/runs/:runId" element={<RunView />} />
        {/* Permalinks to individual recorded games (kept forever on disk). */}
        <Route path="/runs/:runId/games/:gen/:idx" element={<GameView kind="selfplay" />} />
        <Route path="/runs/:runId/eval/:seq" element={<GameView kind="eval" />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}
