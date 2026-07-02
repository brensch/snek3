import { Navigate, Route, Routes } from "react-router-dom";
import { BenchView } from "./pages/BenchView";
import { RunView } from "./pages/RunView";
import { RunsHome } from "./pages/RunsHome";

export default function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      <Routes>
        <Route path="/" element={<RunsHome />} />
        <Route path="/bench" element={<BenchView />} />
        <Route path="/runs/:runId" element={<RunView />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}
