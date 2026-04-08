const EVENT_COPY = {
  power_outage: "Power outage: ventilator and critical-care capacity reduced.",
  ambulance_surge: "Ambulance surge: inbound patient arrivals have spiked sharply.",
  doctor_unavailability: "Doctor unavailability: staffing capacity temporarily constrained.",
  icu_overload: "ICU overload: downstream critical-care flow is impaired.",
};

export function EventBanner({ events = [] }) {
  if (!events.length) {
    return (
      <div className="glass-panel rounded-[24px] border border-emerald-100 px-5 py-4 shadow-panel">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-emerald-100 text-xl">
            ✓
          </div>
          <div>
            <p className="font-semibold text-ink">Operations stable</p>
            <p className="text-sm text-slate-500">
              No dynamic disruptions are active right now.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="relative overflow-hidden rounded-[24px] border border-red-200 bg-gradient-to-r from-red-500 to-red-400 px-5 py-4 text-white shadow-panel">
      <div className="absolute inset-0 bg-white/10" />
      <div className="absolute inset-y-0 left-0 w-1/3 bg-white/15 blur-2xl" />
      <div className="absolute top-0 h-full w-24 bg-white/20 opacity-60 animate-sweep" />
      <div className="relative z-10 flex flex-wrap gap-4">
        {events.map((event) => {
          const type = event.event_type || event.type || "alert";
          return (
            <div key={`${type}-${event.start_step || 0}`} className="min-w-[220px] flex-1">
              <p className="text-xs font-semibold uppercase tracking-[0.28em] text-red-100">
                Active Alert
              </p>
              <p className="mt-1 font-display text-lg capitalize">{type.replaceAll("_", " ")}</p>
              <p className="mt-1 text-sm text-red-50/95">
                {EVENT_COPY[type] || event.description || "Hospital operations are under stress."}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
