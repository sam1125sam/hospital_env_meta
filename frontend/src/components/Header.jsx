function StatChip({ label, value, accent = "bg-clinical-100 text-clinical-800" }) {
  return (
    <div className={`rounded-2xl px-4 py-3 ${accent}`}>
      <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-500">
        {label}
      </p>
      <p className="mt-1 text-xl font-bold text-ink">{value}</p>
    </div>
  );
}

export function Header({ summary, status, config }) {
  return (
    <header className="relative overflow-hidden rounded-[32px] border border-white/70 bg-gradient-to-br from-clinical-800 via-clinical-700 to-sky-700 px-6 py-6 text-white shadow-glow">
      <div className="absolute inset-0 bg-clinical-grid bg-[size:28px_28px] opacity-15" />
      <div className="absolute inset-y-0 left-[-20%] w-1/2 rotate-12 bg-white/10 blur-3xl" />
      <div className="relative z-10 grid gap-6 lg:grid-cols-[1.5fr_1fr]">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.4em] text-clinical-100/80">
            AI Emergency Response Command
          </p>
          <h1 className="mt-3 max-w-3xl font-display text-3xl leading-tight sm:text-4xl">
            Real-time hospital crisis orchestration for ethical, multi-agent triage.
          </h1>
          <p className="mt-4 max-w-2xl text-sm text-clinical-50/85 sm:text-base">
            Monitor live queue pressure, doctor decisions, dynamic disruptions, and
            outcome metrics as the simulation evolves.
          </p>
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <StatChip label="Difficulty" value={config?.difficulty || "medium"} accent="bg-white/12 text-white" />
          <StatChip label="Ethical Mode" value={config?.ethical_mode || "fairness"} accent="bg-white/12 text-white" />
          <StatChip label="Queue" value={summary?.queue_length ?? 0} accent="bg-white/12 text-white" />
          <StatChip
            label="Step"
            value={`${status?.episode_step ?? 0}`}
            accent={status?.done ? "bg-red-400/20 text-white" : "bg-white/12 text-white"}
          />
        </div>
      </div>
    </header>
  );
}
