import { ConfigPanel } from "./components/ConfigPanel";
import { GenerationChart } from "./components/GenerationChart";
import { Header } from "./components/Header";
import { MetricChart } from "./components/MetricChart";
import { RunControls } from "./components/RunControls";
import { StatGrid } from "./components/StatGrid";
import { useTrainer } from "./hooks/useTrainer";

export default function App() {
  const { config, runs, state, stats, history, generationHistory, actions } = useTrainer();
  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      <Header runs={runs} state={state} />
      <main className="mx-auto grid max-w-7xl gap-4 p-4">
        <RunControls runs={runs} state={state} onStart={actions.start} onStop={actions.stop} />
        <StatGrid stats={stats} state={state} />
        <section className="grid gap-3 lg:grid-cols-3">
          <MetricChart rows={history} field="inferences_per_sec" label="Inference rate" />
          <MetricChart rows={history} field="gpu_rows_per_sec" label="GPU rows/s" />
          <MetricChart rows={history} field="games_per_sec" label="Game rate" />
          <MetricChart rows={history} field="gpu_busy_pct" label="GPU busy" />
        </section>
        <section className="grid gap-3 lg:grid-cols-2">
          <GenerationChart rows={generationHistory} field="policy_loss" label="Policy loss by generation" />
          <GenerationChart rows={generationHistory} field="value_loss" label="Value loss by generation" />
        </section>
        <ConfigPanel config={config} onSave={actions.saveConfig} />
      </main>
    </div>
  );
}
