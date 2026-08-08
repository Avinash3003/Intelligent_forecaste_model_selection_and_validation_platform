import { cn } from '../../utils/cn'

/**
 * Premium range slider with an optional tick scale beneath the track.
 *
 * The track and its filled portion are plain divs so they theme with
 * Tailwind; the native input sits transparently on top and supplies the
 * draggable thumb (styled in styles/index.css) plus full keyboard and
 * accessibility support for free.
 *
 * Props:
 *   value, min, max, step  — standard range semantics
 *   ticks                  — [{ value, label }] rendered under the track
 *   valueLabel / hint      — headline + secondary readout above the track
 *   formatBound            — formats the min/max end labels
 */
export default function RangeSlider({
  value,
  min,
  max,
  step = 1,
  onChange,
  ticks = [],
  valueLabel,
  hint,
  formatBound = (v) => v,
  className,
}) {
  const percentFor = (v) => ((v - min) / (max - min)) * 100
  const percent = percentFor(value)

  return (
    <div className={cn('w-full', className)}>
      {(valueLabel || hint) && (
        <div className="mb-5 flex items-baseline gap-2.5">
          <span className="text-2xl font-bold tracking-tight text-slate-800 dark:text-slate-100">
            {valueLabel}
          </span>
          {hint && <span className="text-sm font-medium text-slate-400">{hint}</span>}
        </div>
      )}

      <div className="relative flex h-5 items-center">
        {/* Unfilled track */}
        <div className="pointer-events-none absolute inset-x-0 h-2 rounded-full bg-slate-200 dark:bg-slate-700" />

        {/* Filled portion up to the current value */}
        <div
          className="pointer-events-none absolute h-2 rounded-full bg-gradient-to-r from-brand-500 to-indigo-600"
          style={{ width: `${percent}%` }}
        />

        {/* Tick markers sit on the track itself for precise alignment */}
        {ticks.map((tick) => (
          <span
            key={tick.value}
            className={cn(
              'pointer-events-none absolute h-2 w-px -translate-x-1/2',
              tick.value <= value ? 'bg-white/50' : 'bg-slate-300 dark:bg-slate-600'
            )}
            style={{ left: `${percentFor(tick.value)}%` }}
          />
        ))}

        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(event) => onChange?.(Number(event.target.value))}
          className="range-slider relative z-10 w-full"
          aria-valuetext={valueLabel}
        />
      </div>

      {ticks.length > 0 && (
        <div className="relative mt-2 h-5">
          {ticks.map((tick) => (
            <button
              key={tick.value}
              type="button"
              onClick={() => onChange?.(tick.value)}
              style={{ left: `${percentFor(tick.value)}%` }}
              className={cn(
                'absolute -translate-x-1/2 rounded px-1 text-xs font-medium transition-colors',
                tick.value === value
                  ? 'text-brand-600 dark:text-brand-400'
                  : 'text-slate-400 hover:text-slate-600 dark:hover:text-slate-300'
              )}
            >
              {tick.label}
            </button>
          ))}
        </div>
      )}

      <div className="mt-1 flex justify-between text-xs text-slate-400">
        <span>{formatBound(min)}</span>
        <span>{formatBound(max)}</span>
      </div>
    </div>
  )
}
