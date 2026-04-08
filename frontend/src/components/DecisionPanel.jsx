import { SectionCard } from "./SectionCard";

export function DecisionPanel({ actions = [], conflicts = [], reasoning = "", ethicalMode = "" }) {
  return (
    <SectionCard
      title="Agent Decisions"
      subtitle="Live doctor actions, resolved targets, and conflict handling."
      action={
        <span className="rounded-full bg-clinical-100 px-3 py-1 text-xs font-bold uppercase tracking-[0.18em] text-clinical-700">
          {ethicalMode}
        </span>
      }
      className="h-full"
    >
      <div className="grid gap-3">
        <div className="rounded-[22px] border border-slate-200 bg-slate-50 p-4">
          <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">
            Decision Logic
          </p>
          <p className="mt-2 text-sm text-slate-700">{reasoning || "Awaiting first step."}</p>
        </div>

        <div className="grid gap-3">
          {actions.map((action) => {
            const requested = action.requested?.action_type || "wait";
            const resolved = action.resolved?.action_type || "wait";
            const patientId = action.resolved?.patient_id || "none";
            return (
              <div
                key={action.agent_id}
                className={`rounded-[22px] border p-4 transition ${
                  action.started_treatment
                    ? "border-emerald-200 bg-emerald-50/70"
                    : action.success
                      ? "border-clinical-200 bg-clinical-50/60"
                      : "border-red-200 bg-red-50/70"
                }`}
              >
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="font-semibold text-ink">Doctor Agent {action.agent_id + 1}</p>
                    <p className="text-xs uppercase tracking-[0.18em] text-slate-500">
                      Requested {requested.replaceAll("_", " ")}
                    </p>
                  </div>
                  <span className="rounded-full bg-white px-3 py-1 text-xs font-bold text-slate-700">
                    Reward {Number(action.reward || 0).toFixed(2)}
                  </span>
                </div>
                <div className="mt-3 grid gap-2 text-sm text-slate-700">
                  <p>
                    <span className="font-semibold text-slate-900">Resolved action:</span>{" "}
                    {resolved.replaceAll("_", " ")}
                  </p>
                  <p>
                    <span className="font-semibold text-slate-900">Selected patient:</span> {patientId}
                  </p>
                  <p>
                    <span className="font-semibold text-slate-900">Explanation:</span> {action.reason}
                  </p>
                </div>
              </div>
            );
          })}
        </div>

        {conflicts.length ? (
          <div className="rounded-[22px] border border-amber-200 bg-amber-50 p-4">
            <p className="text-xs font-semibold uppercase tracking-[0.24em] text-amber-700">
              Agent Conflicts
            </p>
            <div className="mt-3 grid gap-3">
              {conflicts.map((conflict, index) => (
                <div key={`${conflict.patient_id}-${index}`} className="rounded-2xl bg-white px-4 py-3 text-sm text-slate-700">
                  <p className="font-semibold text-ink">
                    Patient {conflict.patient_id} contested by agents {conflict.competing_agents.join(", ")}
                  </p>
                  <p className="mt-1">
                    Winner: Agent {conflict.winner}. {conflict.reason}
                  </p>
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </div>
    </SectionCard>
  );
}
