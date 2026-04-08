import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { SectionCard } from "./SectionCard";

export function MetricsPanel({ history = [], summary = {} }) {
  return (
    <SectionCard
      title="Metrics & Analytics"
      subtitle="Outcome performance, fairness, mortality, and utilization over time."
      className="h-full"
    >
      <div className="grid gap-4">
        <div className="grid gap-3 sm:grid-cols-4">
          <div className="rounded-[22px] bg-slate-50 p-4">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Survival Rate</p>
            <p className="mt-2 font-display text-2xl text-ink">
              {Math.round((summary.survival_rate || 0) * 100)}%
            </p>
          </div>
          <div className="rounded-[22px] bg-slate-50 p-4">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Deaths</p>
            <p className="mt-2 font-display text-2xl text-ink">{summary.deaths || 0}</p>
          </div>
          <div className="rounded-[22px] bg-slate-50 p-4">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Fairness Score</p>
            <p className="mt-2 font-display text-2xl text-ink">
              {Math.round((summary.fairness_score || 0) * 100)}%
            </p>
          </div>
          <div className="rounded-[22px] bg-slate-50 p-4">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Utilization</p>
            <p className="mt-2 font-display text-2xl text-ink">
              {Math.round((summary.resource_utilization || 0) * 100)}%
            </p>
          </div>
        </div>

        <div className="grid gap-4 xl:grid-cols-2">
          <div className="rounded-[24px] border border-slate-200 bg-white p-4">
            <p className="mb-4 font-semibold text-ink">Survival and Fairness Trajectory</p>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={history}>
                  <CartesianGrid stroke="#d9e6ee" strokeDasharray="3 3" />
                  <XAxis dataKey="step" stroke="#7b91a3" />
                  <YAxis stroke="#7b91a3" />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="survival_rate" stroke="#2a7fa8" strokeWidth={3} dot={false} name="Survival Rate" />
                  <Line type="monotone" dataKey="fairness_score" stroke="#1ea672" strokeWidth={3} dot={false} name="Fairness Score" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="rounded-[24px] border border-slate-200 bg-white p-4">
            <p className="mb-4 font-semibold text-ink">Mortality and Resource Load</p>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={history}>
                  <CartesianGrid stroke="#d9e6ee" strokeDasharray="3 3" />
                  <XAxis dataKey="step" stroke="#7b91a3" />
                  <YAxis stroke="#7b91a3" />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="deaths" fill="#d94141" radius={[8, 8, 0, 0]} name="Deaths" />
                  <Bar dataKey="resource_utilization" fill="#66b3d2" radius={[8, 8, 0, 0]} name="Utilization" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </SectionCard>
  );
}
