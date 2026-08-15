import {
  ResponsiveContainer,
  ComposedChart,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  Line,
  Area,
  Brush,
  ReferenceLine,
} from 'recharts'
import SectionContainer from '../../../components/layout/SectionContainer'
import ResultsFilterBar from './ResultsFilterBar'

function ForecastDot(props) {
  const { cx, cy, payload } = props
  if (payload.forecast == null) return null

  if (payload.highlight) {
    return (
      <g>
        <circle cx={cx} cy={cy} r={7} fill="#4f46e5" stroke="white" strokeWidth={2} />
        <circle cx={cx} cy={cy} r={12} fill="#4f46e5" fillOpacity={0.15} />
      </g>
    )
  }
  return <circle cx={cx} cy={cy} r={3} fill="#818cf8" />
}

// The backend sends actual points as real dates ("2015-01-01") and forecast
// horizon points as "T1", "T2", ... — never mixed up, so this is a safe way
// to tell which formatting rule applies to a given axis value.
function isDateLike(period) {
  return typeof period === 'string' && /^\d{4}-\d{2}-\d{2}/.test(period)
}

function parseDatePeriod(period) {
  const date = new Date(period.slice(0, 10))
  return Number.isNaN(date.getTime()) ? null : date
}

// Format granularity follows the *actual* span present in the data, never
// a fixed assumption about the dataset's grain: a two-month upload reads
// naturally as "Jan 5", a multi-decade one as just the year, and anything
// in between as "Jan 24" — the same "no unnecessary precision" rule
// Data Coverage (Priority #8) already applies, extended to per-tick
// granularity here.
function formatPeriodTick(period, spanDays) {
  if (!isDateLike(period)) return period
  const date = parseDatePeriod(period)
  if (!date) return period
  if (spanDays > 365 * 3) return date.toLocaleDateString('en-US', { year: 'numeric' })
  if (spanDays > 60) return date.toLocaleDateString('en-US', { month: 'short', year: '2-digit' })
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

function formatTooltipLabel(period) {
  if (!isDateLike(period)) return period
  const date = parseDatePeriod(period)
  return date ? date.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' }) : period
}

function formatValue(value) {
  if (value == null) return '—'
  return Number(value).toLocaleString('en-US', { maximumFractionDigits: 2 })
}

// One tooltip for the whole chart, so the forecast interval reads as a
// single "range" row rather than as two unexplained series called
// forecastLower/forecastUpper.
function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  const point = payload[0]?.payload
  if (!point) return null

  return (
    <div className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs shadow-lg dark:border-slate-700 dark:bg-slate-900">
      <p className="mb-1 font-semibold text-slate-700 dark:text-slate-200">
        {formatTooltipLabel(label)}
      </p>
      {point.actual != null && (
        <p className="flex items-center gap-2 text-slate-500 dark:text-slate-400">
          <span className="inline-block h-2 w-2 rounded-full bg-slate-900 dark:bg-slate-100" />
          Actual <span className="ml-auto font-semibold tabular-nums">{formatValue(point.actual)}</span>
        </p>
      )}
      {point.forecast != null && (
        <p className="flex items-center gap-2 text-slate-500 dark:text-slate-400">
          <span className="inline-block h-2 w-2 rounded-full bg-indigo-500" />
          Forecast <span className="ml-auto font-semibold tabular-nums">{formatValue(point.forecast)}</span>
        </p>
      )}
      {point.lower != null && point.upper != null && (
        <p className="mt-0.5 text-[11px] text-slate-400">
          Range {formatValue(point.lower)} – {formatValue(point.upper)}
        </p>
      )}
    </div>
  )
}

// How many axis ticks stay legible before they start overlapping — chosen
// once, applied via recharts' own `interval` (which only skips *label*
// rendering; every data point is still plotted and connected).
const MAX_VISIBLE_TICKS = 10
// Above this many total points, individual actual-line dot markers start
// reading as visual noise rather than data — the line itself still shows
// every point, this only turns off the per-point marker.
const DOT_DENSITY_THRESHOLD = 60
// Above this many points the series is too dense to read end to end at
// card width, so a brush is offered to scrub into any sub-range. Below it
// the whole series already fits comfortably and a brush is just clutter.
const BRUSH_THRESHOLD = 120

export default function ActualVsForecastChart({ filters, points = [] }) {
  // The selected horizon point is highlighted on the forecast line; every
  // other point stays visible but secondary (Section 5.6).
  //
  // `band` is recharts' range-area shape ([low, high]) built from the
  // forecast interval the backend already sends per point. It was being
  // sent and silently discarded, which left the forecast drawn as a bare
  // line with no visible uncertainty at all.
  const data = points.map((point) => ({
    ...point,
    highlight: point.period === filters.horizon,
    band: point.lower != null && point.upper != null ? [point.lower, point.upper] : null,
  }))

  // Derived from the real data, not assumed: the actual-history points'
  // own date span decides both the tick date format and how many labels
  // can fit without overlapping. No points are ever dropped — only which
  // of them get an axis label is thinned.
  const actualDates = data
    .filter((point) => point.actual != null)
    .map((point) => parseDatePeriod(point.period))
    .filter(Boolean)
  const spanDays =
    actualDates.length > 1
      ? (Math.max(...actualDates) - Math.min(...actualDates)) / (1000 * 60 * 60 * 24)
      : 0
  const tickInterval = data.length > MAX_VISIBLE_TICKS ? Math.ceil(data.length / MAX_VISIBLE_TICKS) - 1 : 0
  const showDots = data.length <= DOT_DENSITY_THRESHOLD
  const showBrush = data.length > BRUSH_THRESHOLD

  // Where history ends and the forecast begins — marked once so a long
  // series still reads as "past | future" at a glance instead of relying
  // on the reader spotting where the line turns dashed.
  const forecastStart = data.find((point) => point.forecast != null && point.actual == null)?.period

  // A brush needs its own vertical room; without it the plot area would be
  // squeezed rather than the card growing to fit.
  const plotHeight = showBrush ? 'h-[26rem]' : 'h-80'

  return (
    <SectionContainer
      title="Actual vs Forecast"
      subtitle="Complete historical actuals plus the forward forecast, with the selected horizon point highlighted"
    >
      {/* Filters sit directly above the chart they drive, rather than in a
          detached bar at the top of the page. */}
      <div className="mb-6 border-b border-slate-100 pb-5 dark:border-slate-800">
        <ResultsFilterBar {...filters} />
      </div>

      <div className={`w-full ${plotHeight}`}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 10, right: 16, left: 4, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-slate-100 dark:text-slate-800" />
            <XAxis
              dataKey="period"
              tick={{ fontSize: 12, fill: '#94a3b8' }}
              axisLine={false}
              tickLine={false}
              interval={tickInterval}
              minTickGap={20}
              tickFormatter={(value) => formatPeriodTick(value, spanDays)}
              padding={{ left: 12, right: 12 }}
            />
            <YAxis
              tick={{ fontSize: 12, fill: '#94a3b8' }}
              axisLine={false}
              tickLine={false}
              domain={['auto', 'auto']}
              width={48}
            />
            <Tooltip content={<ChartTooltip />} />
            <Legend wrapperStyle={{ fontSize: 12 }} />

            {/* Drawn before the lines so the shaded interval sits behind
                them rather than washing them out. */}
            <Area
              type="monotone"
              dataKey="band"
              name="Forecast range"
              stroke="none"
              fill="#4f46e5"
              fillOpacity={0.12}
              connectNulls
              isAnimationActive={false}
              legendType="none"
            />

            {forecastStart && (
              <ReferenceLine
                x={forecastStart}
                stroke="#94a3b8"
                strokeDasharray="4 4"
                label={{
                  value: 'Forecast',
                  position: 'insideTopRight',
                  fontSize: 10,
                  fill: '#94a3b8',
                }}
              />
            )}

            <Line
              type="monotone"
              dataKey="actual"
              name="Actual"
              stroke="#0f172a"
              strokeWidth={2}
              dot={showDots ? { r: 3 } : false}
              activeDot={{ r: 5 }}
              connectNulls
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="forecast"
              name="Forecast"
              stroke="#4f46e5"
              strokeWidth={2}
              strokeDasharray="6 4"
              dot={<ForecastDot />}
              connectNulls
              isAnimationActive={false}
            />

            {/* Only for series too long to read whole: drag to scrub into
                any sub-range. The underlying data is untouched — this
                changes what is on screen, never what was plotted. */}
            {showBrush && (
              <Brush
                dataKey="period"
                height={26}
                travellerWidth={8}
                stroke="#c7d2fe"
                fill="#f8fafc"
                tickFormatter={(value) => formatPeriodTick(value, spanDays)}
                startIndex={Math.max(0, data.length - 120)}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </SectionContainer>
  )
}
