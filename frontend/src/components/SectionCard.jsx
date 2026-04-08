export function SectionCard({ title, subtitle, action, children, className = "" }) {
  return (
    <section
      className={`glass-panel rounded-[28px] border border-white/70 p-5 shadow-panel ${className}`}
    >
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <p className="font-display text-lg text-ink">{title}</p>
          {subtitle ? <p className="mt-1 text-sm text-slate-500">{subtitle}</p> : null}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}
