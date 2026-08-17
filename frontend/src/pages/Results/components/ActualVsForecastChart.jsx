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
import {
  chooseGranularity,
  formatLabel,
  formatTooltipLabel,
  historySpanDays,
  pointLabel,
  selectTicks,
} from './chartAxis'

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

function formatValue(value) {
  if (value == null) return '—'
  return Number(value).toLocaleString('en-US', { maximumFractionDigits: 2 })
}

// One tooltip for the whole chart, so the forecast interval reads as a
// single "range" row rather than as two unexplained series called
// forecastLower/forecastUpper.
function ChartTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const point = payload[0]?.payload
  if (!point) return null

  return (
    <div className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs shadow-lg dark:border-slate-700 dark:bg-slate-900">
      <p className="mb-1 font-semibold text-slate-700 dark:text-slate-200">
        {formatTooltipLabel(pointLabel(point))}
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

// How many axis ticks stay legible at card width before they overlap.
// Applied to *labels* only — every data point is plotted and connected
// regardless of how few ticks survive.
const MAX_VISIBLE_TICKS = 9
// Above this many total points, per-point dot markers read as noise
// rather than as data. The line still contains every point.
const DOT_DENSITY_THRESHOLD = 60
// Above this many points the series cannot be read end to end at card
// width, so a brush is offered to scrub into any sub-range.
const BRUSH_THRESHOLD = 120
// How much of the tail the brush opens on. The whole series stays
// scrubbable — this is only the initial window.
const BRUSH_INITIAL_WINDOW = 120

export default function ActualVsForecastChart({ filters, points = [] }) {
  // The selected horizon point is highlighted on the forecast line; every
  // other point stays visible but secondary.
  //
  // `band` is recharts' range-area shape ([low, high]) built from the
  // forecast interval the backend sends per point — present for models
  // that produce one (ARIMA, Prophet), absent for those that do not (the
  // tree models), which is why it is nulled rather than defaulted.
  const data = points.map((point) => ({
    ...point,
    highlight: point.period === filters.horizon,
    band: point.lower != null && point.upper != null ? [point.lower, point.upper] : null,
  }))

  // Every axis decision below is derived from the data itself — the span
  // it covers and how many points it holds — never from a fixed
  // assumption about grain or length.
  const spanDays = historySpanDays(data)
  const granularity = chooseGranularity(spanDays)
  const ticks = selectTicks(data, granularity, MAX_VISIBLE_TICKS)
  const showDots = data.length <= DOT_DENSITY_THRESHOLD
  const showBrush = data.length > BRUSH_THRESHOLD

  // Where observed history ends. Marked by the backend on the real final
  // historical timestamp, so this is never a position guess.
  const boundaryPeriod = data.find((point) => point.boundary)?.period

  const labelByPeriod = new Map(data.map((point) => [point.period, pointLabel(point)]))

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

      {/* min-w-0 lets the flex/grid parent shrink this below its content
          width, which is what keeps a long series from pushing the card
          into horizontal overflow. */}
      <div className={`w-full min-w-0 ${plotHeight}`}>
        <ResponsiveContainer width="100%" height="100%">
          {/* Margins leave room for the first/last tick label and the
              boundary caption instead of letting them clip at the edges;
              the y-axis reserves its own width rather than borrowing from
              the plot. */}
          <ComposedChart data={data} margin={{ top: 16, right: 24, left: 8, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-slate-100 dark:text-slate-800" />
            <XAxis
              dataKey="period"
              tick={{ fontSize: 12, fill: '#94a3b8' }}
              axisLine={false}
              tickLine={false}
              // Explicit ticks, so which labels appear is decided by
              // selectTicks (distinct + evenly spread) rather than by a
              // blind positional interval.
              ticks={ticks}
              interval={0}
              tickMargin={10}
              minTickGap={16}
              tickFormatter={(value) => formatLabel(labelByPeriod.get(value) ?? value, granularity)}
              padding={{ left: 16, right: 16 }}
            />
            <YAxis
              tick={{ fontSize: 12, fill: '#94a3b8' }}
              axisLine={false}
              tickLine={false}
              domain={['auto', 'auto']}
              width={56}
              tickMargin={8}
              tickFormatter={formatValue}
            />
            <Tooltip content={<ChartTooltip />} />
            <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />

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

            {boundaryPeriod && (
              <ReferenceLine
                x={boundaryPeriod}
                stroke="#94a3b8"
                strokeDasharray="4 4"
                label={{
                  value: 'Forecast →',
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
                tickFormatter={(value) => formatLabel(labelByPeriod.get(value) ?? value, granularity)}
                startIndex={Math.max(0, data.length - BRUSH_INITIAL_WINDOW)}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </SectionContainer>
  )
}
