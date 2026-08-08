export default function SectionHeader({ title, subtitle, action }) {
  return (
    <div className="flex items-center justify-between gap-4 mb-4">
      <div>
        <h2 className="text-base font-semibold text-slate-800 dark:text-slate-100">{title}</h2>
        {subtitle && <p className="text-sm text-slate-400 mt-0.5">{subtitle}</p>}
      </div>
      {action}
    </div>
  )
}
