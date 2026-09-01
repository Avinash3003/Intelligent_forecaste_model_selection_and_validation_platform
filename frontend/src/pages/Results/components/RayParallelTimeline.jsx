import { useMemo, useState } from 'react'
import { ChevronDown, ChevronRight, Cpu } from 'lucide-react'
import { cn } from '../../../utils/cn'

// One lane per Ray worker, one bar per forecast key, positioned by the real
// start/end offsets the stage's own executor recorded (stage.parallelTasks).
//
// Deliberately plain CSS: seven-ish lanes of absolutely-positioned divs is
// not a charting problem, and a chart dependency would be several hundred
// kilobytes for something percentage widths already express exactly.
//
// This is the one view that *proves* key-level parallelism rather than
// asserting it: bars overlapping across different worker lanes is what
// parallel execution looks like, and a single lane with sequential bars is
// what its absence looks like. Neither can be faked by the layout.
//
// v3: one instance per Ray-parallel STAGE (Train/Evaluate/Explain/Rank &
// Select), each fed that stage's own real fan-out — not one timeline for
// the whole run pretending every phase shared a single task. A run with
// four Ray stages shows four of these, each honest about just its own.

// Fixed categorical order, cycled by lane index — identity here is the
// worker, and a worker is otherwise arbitrary, so this exists only to tell
// one lane's bars apart from its neighbour's at a glance.
const LANE_BAR_COLORS = [
  'bg-brand-600',
  'bg-emerald-600',
  'bg-amber-600',
  'bg-sky-600',
  'bg-violet-600',
  'bg-rose-600',
]

const LABEL_COL = 84 // px — the fixed left column both the lanes and the axis align to

function Stat({ label, value }) {
  return (
    <div className="min-w-0">
      <p className="truncate text-[11px] uppercase tracking-wide text-slate-400">{label}</p>
      <p className="mt-0.5 truncate text-sm font-medium text-slate-700 dark:text-slate-200">{value}</p>
    </div>
  )
}

function formatSeconds(value) {
  return `${value.toFixed(value < 10 ? 2 : 1)}s`
}

// stageLabel: the phase name ("Train Models", ...) this fan-out belongs to.
// parallelTasks: one stage's real telemetry — { executor, total, completed,
// failed, running, maxConcurrent, tasks: [{groupId, status, durationSeconds,
// workerId, nodeId, start, end}] }.
export default function RayParallelTimeline({ stageLabel, parallelTasks }) {
  const [open, setOpen] = useState(false)

  const tasks = parallelTasks?.tasks
  const completedTasks = useMemo(() => (tasks ?? []).filter((t) => t.status === 'Completed' && t.start != null), [tasks])
  const failedTasks = useMemo(() => (tasks ?? []).filter((t) => t.status !== 'Completed'), [tasks])
  const hasTasks = parallelTasks?.executor === 'ray' && Array.isArray(tasks) && tasks.length > 0

  // Stable ordinal numbering: sorted by the raw id rather than order of
  // appearance, so "Worker 1" refers to the same lane on every render.
  const workers = useMemo(() => {
    if (!hasTasks) return []
    return [...new Set(completedTasks.map((t) => t.workerId).filter(Boolean))].sort()
  }, [hasTasks, completedTasks])

  const scaleEnd = useMemo(() => {
    if (!completedTasks.length) return 0.001
    return Math.max(...completedTasks.map((t) => t.end), 0.001)
  }, [completedTasks])

  if (!hasTasks) return null

  return (
    <div className="mt-3 rounded-lg border border-slate-200 dark:border-slate-700">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs font-medium text-brand-600 hover:text-brand-700 dark:text-brand-400"
      >
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <Cpu size={13} />
        {stageLabel} — Ray parallel execution — {parallelTasks.total} key
        {parallelTasks.total === 1 ? '' : 's'} across {workers.length} worker
        {workers.length === 1 ? '' : 's'}
      </button>

      {open && (
        <div className="border-t border-slate-100 px-3 py-3 dark:border-slate-800">
          <div className="mb-5 grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
            <Stat label="Executor" value="Ray" />
            <Stat label="Completed" value={`${parallelTasks.completed} / ${parallelTasks.total}`} />
            <Stat label="Max concurrent" value={parallelTasks.maxConcurrent ?? '—'} />
            <Stat
              label="Wall time"
              value={scaleEnd > 0.001 ? formatSeconds(scaleEnd) : '—'}
            />
          </div>

          <div className="relative space-y-1.5">
            {workers.map((workerId, laneIndex) => {
              const laneTasks = completedTasks
                .filter((t) => t.workerId === workerId)
                .sort((a, b) => a.start - b.start)
              const barColor = LANE_BAR_COLORS[laneIndex % LANE_BAR_COLORS.length]

              return (
                <div
                  key={workerId}
                  className="grid items-center gap-2"
                  style={{ gridTemplateColumns: `${LABEL_COL}px 1fr` }}
                >
                  <span
                    className="truncate text-xs font-medium text-slate-500 dark:text-slate-400"
                    title={`Ray worker id ${workerId}`}
                  >
                    Worker {laneIndex + 1}
                  </span>
                  <div className="relative h-6 rounded bg-slate-100 dark:bg-slate-800">
                    {laneTasks.map((task) => {
                      const left = (task.start / scaleEnd) * 100
                      // A key faster than the scale's resolution still has
                      // to be visible, so bars have a floor width.
                      const widthPct = Math.max(((task.end - task.start) / scaleEnd) * 100, 0.6)
                      // Below this, the bar has room for a color and a
                      // hover target but not for its own name — better to
                      // show nothing than three illegible characters.
                      const showLabel = widthPct >= 6

                      return (
                        // "group" scopes the tooltip's group-hover below to
                        // this one bar via pure CSS — no hover state, so a
                        // bar a few pixels wide never has to share those
                        // pixels with a highlight ring; hover just brightens
                        // and lifts it instead of redrawing its border.
                        <div
                          key={task.groupId}
                          className={cn(
                            'group absolute top-1 flex h-4 cursor-default items-center overflow-visible rounded-sm px-1',
                            'transition-[filter] hover:z-10 hover:brightness-110',
                            barColor
                          )}
                          style={{ left: `${left}%`, width: `${widthPct}%` }}
                        >
                          <span className="overflow-hidden text-ellipsis whitespace-nowrap">
                            {showLabel && (
                              <span className="text-[9px] font-medium text-white">{task.groupId}</span>
                            )}
                          </span>

                          <div
                            className="pointer-events-none absolute bottom-full left-1/2 z-20 mb-1.5 hidden w-max max-w-[240px] -translate-x-1/2 rounded-md border border-slate-200 bg-white px-2.5 py-1.5 text-left shadow-lg group-hover:block dark:border-slate-700 dark:bg-slate-800"
                            role="tooltip"
                          >
                            <p className="truncate text-[11px] font-semibold text-slate-800 dark:text-slate-100">
                              {task.groupId}
                            </p>
                            <p className="mt-0.5 text-[10px] text-slate-500 dark:text-slate-400">
                              Worker {laneIndex + 1} · {formatSeconds(task.start)}–{formatSeconds(task.end)} ·{' '}
                              {formatSeconds(task.end - task.start)} duration
                            </p>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )
            })}

            <div className="grid items-center gap-2" style={{ gridTemplateColumns: `${LABEL_COL}px 1fr` }}>
              <span />
              <div className="flex justify-between text-[10px] text-slate-400">
                <span>0s</span>
                <span>{scaleEnd.toFixed(1)}s</span>
              </div>
            </div>
          </div>

          {failedTasks.length > 0 && (
            <p className="mt-3 text-xs text-rose-600 dark:text-rose-400">
              {failedTasks.length} key{failedTasks.length === 1 ? '' : 's'} failed and{' '}
              {failedTasks.length === 1 ? 'is' : 'are'} not drawn above:{' '}
              {failedTasks.map((t) => t.groupId).join(', ')}
            </p>
          )}
        </div>
      )}
    </div>
  )
}
