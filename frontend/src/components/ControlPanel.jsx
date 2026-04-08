import { SectionCard } from "./SectionCard";

function Toggle({ checked, onChange, label }) {
  return (
    <label className="flex items-center justify-between rounded-2xl border border-slate-200 px-4 py-3">
      <span className="text-sm font-semibold text-slate-700">{label}</span>
      <button
        type="button"
        onClick={() => onChange(!checked)}
        className={`relative h-7 w-12 rounded-full transition ${
          checked ? "bg-clinical-500" : "bg-slate-300"
        }`}
      >
        <span
          className={`absolute top-1 h-5 w-5 rounded-full bg-white transition ${
            checked ? "left-6" : "left-1"
          }`}
        />
      </button>
    </label>
  );
}

export function ControlPanel({
  controls,
  options,
  busy,
  isAutoRunning,
  onChange,
  onReset,
  onStep,
  onRunFull,
  onPause,
}) {
  return (
    <SectionCard
      title="Simulation Controls"
      subtitle="Tune the scenario, trigger steps, and drive the full demo flow."
    >
      <div className="grid gap-4">
        <div className="grid gap-4 md:grid-cols-3">
          <label className="grid gap-2 text-sm font-semibold text-slate-700">
            Ethical Mode
            <select
              className="rounded-2xl border border-slate-200 bg-white px-4 py-3 outline-none transition focus:border-clinical-400"
              value={controls.ethical_mode}
              onChange={(event) => onChange("ethical_mode", event.target.value)}
            >
              {options.ethicalModes.map((mode) => (
                <option key={mode.value} value={mode.value}>
                  {mode.label}
                </option>
              ))}
            </select>
          </label>

          <label className="grid gap-2 text-sm font-semibold text-slate-700">
            Task Difficulty
            <select
              className="rounded-2xl border border-slate-200 bg-white px-4 py-3 outline-none transition focus:border-clinical-400"
              value={controls.difficulty}
              onChange={(event) => onChange("difficulty", event.target.value)}
            >
              {options.tasks.map((task) => (
                <option key={task.key} value={task.key}>
                  {task.label}
                </option>
              ))}
            </select>
          </label>

          <label className="grid gap-2 text-sm font-semibold text-slate-700">
            Doctor Agents
            <select
              className="rounded-2xl border border-slate-200 bg-white px-4 py-3 outline-none transition focus:border-clinical-400"
              value={controls.num_agents}
              onChange={(event) => onChange("num_agents", Number(event.target.value))}
            >
              {[1, 2, 3, 4, 5, 6].map((count) => (
                <option key={count} value={count}>
                  {count} agents
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="grid gap-3 md:grid-cols-2">
          <Toggle
            checked={controls.uncertainty}
            onChange={(value) => onChange("uncertainty", value)}
            label="Uncertainty Layer"
          />
          <Toggle
            checked={controls.dynamic_events}
            onChange={(value) => onChange("dynamic_events", value)}
            label="Dynamic Events"
          />
        </div>

        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
          <button
            type="button"
            onClick={onReset}
            disabled={busy || isAutoRunning}
            className="rounded-2xl bg-clinical-500 px-4 py-3 text-sm font-semibold text-white transition hover:bg-clinical-600 disabled:opacity-60"
          >
            Start / Reset
          </button>
          <button
            type="button"
            onClick={onStep}
            disabled={busy || isAutoRunning}
            className="rounded-2xl border border-clinical-200 bg-white px-4 py-3 text-sm font-semibold text-clinical-700 transition hover:border-clinical-400"
          >
            Step Forward
          </button>
          <button
            type="button"
            onClick={onRunFull}
            disabled={busy || isAutoRunning}
            className="rounded-2xl bg-ink px-4 py-3 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:opacity-60"
          >
            Run Full Simulation
          </button>
          <button
            type="button"
            onClick={onPause}
            disabled={!isAutoRunning}
            className="rounded-2xl border border-red-200 bg-white px-4 py-3 text-sm font-semibold text-red-600 transition hover:border-red-400 disabled:opacity-60"
          >
            Pause
          </button>
          <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
            <span className="font-semibold text-slate-900">
              {isAutoRunning ? "Live run in progress" : "Manual control mode"}
            </span>
            <p className="mt-1 text-xs text-slate-500">
              Judges can watch the system evolve step by step or end-to-end.
            </p>
          </div>
        </div>
      </div>
    </SectionCard>
  );
}
