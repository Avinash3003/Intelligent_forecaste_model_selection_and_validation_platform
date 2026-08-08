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

export default function ActualVsForecastChart({ filters, points = [] }) {
  // The selected horizon point is highlighted on the forecast line; every
  // other point stays visible but secondary (Section 5.6).
  const data = points.map((point) => ({
    ...point,
    highlight: point.period === filters.horizon,
  }))

  return (
    <SectionContainer
      title="Actual vs Forecast"
      subtitle="12-month forward forecast with the selected horizon point highlighted"
    >
      {/* Filters sit directly above the chart they drive, rather than in a
          detached bar at the top of the page. */}
      <div className="mb-6 border-b border-slate-100 pb-5 dark:border-slate-800">
        <ResultsFilterBar {...filters} />
      </div>

      <div className="h-80 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 10, right: 16, left: -12, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-slate-100 dark:text-slate-800" />
            <XAxis dataKey="period" tick={{ fontSize: 12, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fontSize: 12, fill: '#94a3b8' }} axisLine={false} tickLine={false} domain={['auto', 'auto']} />
            <Tooltip
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
              dot={{ r: 3 }}
              connectNulls
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
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </SectionContainer>
  )
}
