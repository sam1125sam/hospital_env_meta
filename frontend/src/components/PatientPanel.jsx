import { SectionCard } from "./SectionCard";

function severityTone(severity) {
  if (severity >= 7) {
    return "bg-red-50 text-red-700 ring-1 ring-red-200";
  }
  if (severity >= 4) {
    return "bg-amber-50 text-amber-700 ring-1 ring-amber-200";
  }
  return "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200";
}

export function PatientPanel({ patients = [] }) {
  const queue = [...patients].sort((a, b) => {
    if (b.severity !== a.severity) {
      return b.severity - a.severity;
    }
    return b.wait_time - a.wait_time;
  });

  return (
    <SectionCard
      title="Patient Queue"
      subtitle="Severity-aware queue view with survival pressure and wait-time signals."
      className="h-full"
    >
      <div className="overflow-hidden rounded-[22px] border border-slate-200">
        <div className="grid grid-cols-[1.1fr_0.8fr_1fr_0.7fr_1fr] bg-slate-100 px-4 py-3 text-xs font-bold uppercase tracking-[0.18em] text-slate-500">
          <span>Patient</span>
          <span>Severity</span>
          <span>Survival</span>
          <span>Wait</span>
          <span>Condition</span>
        </div>
        <div className="max-h-[460px] overflow-y-auto bg-white">
          {queue.length ? (
            queue.map((patient) => (
              <div
                key={patient.patient_id}
                className={`grid grid-cols-[1.1fr_0.8fr_1fr_0.7fr_1fr] items-center gap-3 border-t border-slate-100 px-4 py-3 text-sm transition ${
                  patient.severity >= 7 ? "bg-red-50/40" : "hover:bg-slate-50"
                }`}
              >
                <div>
                  <p className="font-semibold text-ink">{patient.patient_id}</p>
                  <p className="text-xs text-slate-500 capitalize">{patient.status}</p>
                </div>
                <div>
                  <span className={`inline-flex rounded-full px-3 py-1 text-xs font-bold ${severityTone(patient.severity)}`}>
                    {patient.severity}/10
                  </span>
                </div>
                <div>
                  <p className="font-semibold text-slate-900">
                    {Math.round((patient.survival_probability || 0) * 100)}%
                  </p>
                  <div className="mt-1 h-2 rounded-full bg-slate-100">
                    <div
                      className={`h-2 rounded-full ${
                        patient.survival_probability < 0.4
                          ? "bg-red-500"
                          : patient.survival_probability < 0.7
                            ? "bg-amber-400"
                            : "bg-emerald-500"
                      }`}
                      style={{ width: `${Math.max(8, (patient.survival_probability || 0) * 100)}%` }}
                    />
                  </div>
                </div>
                <div className="font-semibold text-slate-700">{patient.wait_time}</div>
                <div className="capitalize text-slate-600">{String(patient.condition).replaceAll("_", " ")}</div>
              </div>
            ))
          ) : (
            <div className="px-4 py-8 text-sm text-slate-500">
              No waiting patients. The queue is currently clear.
            </div>
          )}
        </div>
      </div>
    </SectionCard>
  );
}
