import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { SectionCard } from "./SectionCard";

export function CompareModesPanel({ data = [], busy, onRefresh }) {
  return (
    <SectionCard
      title="Compare Modes"
      subtitle="Side-by-side benchmark for utilitarian, fairness, and critical-first strategies."
      action={
        <button
          type="button"
          onClick={onRefresh}
          disabled={busy}
          className="rounded-full border border-clinical-200 bg-white px-3 py-2 text-xs font-bold uppercase tracking-[0.18em] text-clinical-700 transition hover:border-clinical-400 disabled:opacity-60"
        >
          Refresh
        </button>
      }
    >
      <div className="grid gap-4 xl:grid-cols-[1.2fr_1fr]">
        <div className="h-80 rounded-[24px] border border-slate-200 bg-white p-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data}>
              <CartesianGrid stroke="#d9e6ee" strokeDasharray="3 3" />
              <XAxis dataKey="label" stroke="#7b91a3" />
              <YAxis stroke="#7b91a3" />
              <Tooltip />
              <Bar dataKey="survival_rate" fill="#2a7fa8" radius={[8, 8, 0, 0]} name="Survival Rate" />
              <Bar dataKey="fairness_score" fill="#1ea672" radius={[8, 8, 0, 0]} name="Fairness Score" />
              <Bar dataKey="resource_utilization" fill="#f0b429" radius={[8, 8, 0, 0]} name="Utilization" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="grid gap-3">
          {data.map((mode) => (
            <div key={mode.mode} className="rounded-[22px] border border-slate-200 bg-white p-4">
              <div className="flex items-center justify-between gap-3">
                <p className="font-display text-lg text-ink">{mode.label}</p>
                <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-bold uppercase tracking-[0.18em] text-slate-600">
                  reward {mode.total_reward}
                </span>
              </div>
              <div className="mt-3 grid grid-cols-2 gap-3 text-sm text-slate-600">
                <div>
                  <p className="text-xs uppercase tracking-[0.18em] text-slate-400">Survival</p>
                  <p className="mt-1 font-semibold text-slate-900">{Math.round(mode.survival_rate * 100)}%</p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-[0.18em] text-slate-400">Fairness</p>
                  <p className="mt-1 font-semibold text-slate-900">{Math.round(mode.fairness_score * 100)}%</p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-[0.18em] text-slate-400">Critical Survival</p>
                  <p className="mt-1 font-semibold text-slate-900">{Math.round(mode.critical_survival_rate * 100)}%</p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-[0.18em] text-slate-400">Avg Wait</p>
                  <p className="mt-1 font-semibold text-slate-900">{mode.avg_wait_time}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </SectionCard>
  );
}
