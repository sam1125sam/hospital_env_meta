import { SectionCard } from "./SectionCard";

function ResourceBar({ label, available, total }) {
  const used = Math.max(0, total - available);
  const percent = total ? (used / total) * 100 : 0;
  const tone =
    percent >= 80 ? "bg-red-500" : percent >= 55 ? "bg-amber-400" : "bg-clinical-500";

  return (
    <div className="rounded-[22px] border border-slate-200 bg-white p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-ink">{label}</p>
          <p className="text-xs text-slate-500">
            {available} available of {total}
          </p>
        </div>
        <div className="metric-ring grid h-14 w-14 place-items-center rounded-full" style={{ "--ring-value": `${Math.round(percent * 3.6)}deg` }}>
          <span className="text-xs font-bold text-ink">{Math.round(percent)}%</span>
        </div>
      </div>
      <div className="mt-4 h-3 overflow-hidden rounded-full bg-slate-100">
        <div className={`h-full rounded-full transition-all duration-300 ${tone}`} style={{ width: `${Math.max(percent, 6)}%` }} />
      </div>
    </div>
  );
}

export function ResourcePanel({ resources = {} }) {
  const cards = [
    { label: "Doctors", key: "doctors" },
    { label: "Beds", key: "beds" },
    { label: "Ventilators", key: "ventilators" },
    { label: "ICU Beds", key: "icu_beds" },
    { label: "OR Rooms", key: "or_rooms" },
  ];

  return (
    <SectionCard
      title="Resource Pressure"
      subtitle="Capacity gauges surface stress before the queue cascades."
      className="h-full"
    >
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-1">
        {cards.map((item) => (
          <ResourceBar
            key={item.key}
            label={item.label}
            available={resources[item.key]?.available ?? 0}
            total={resources[item.key]?.total ?? 0}
          />
        ))}
      </div>
    </SectionCard>
  );
}
