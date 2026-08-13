import {
  ResponsiveContainer,
  ComposedChart,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  Line,
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

// How many axis ticks stay legible before they start overlapping — chosen
// once, applied via recharts' own `interval` (which only skips *label*
// rendering; every data point is still plotted and connected).
const MAX_VISIBLE_TICKS = 10
// Above this many total points, individual actual-line dot markers start
// reading as visual noise rather than data — the line itself still shows
// every point, this only turns off the per-point marker.
const DOT_DENSITY_THRESHOLD = 60

export default function ActualVsForecastChart({ filters, points = [] }) {
  // The selected horizon point is highlighted on the forecast line; every
  // other point stays visible but secondary (Section 5.6).
  const data = points.map((point) => ({
    ...point,
    highlight: point.period === filters.horizon,
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

      <div className="h-80 w-full overflow-hidden">
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
            <Tooltip
              labelFormatter={formatTooltipLabel}
              contentStyle={{
                borderRadius: 12,
                border: '1px solid #e2e8f0',
                fontSize: 12,
                boxShadow: '0 8px 24px rgba(15,23,42,0.12)',
              }}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
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
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </SectionContainer>
  )
}
