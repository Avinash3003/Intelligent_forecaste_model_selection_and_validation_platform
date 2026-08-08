import ExpandablePanel from '../../../components/common/ExpandablePanel'
import Badge from '../../../components/ui/Badge'

const metricFields = [
  { key: 'driftTest', label: 'Drift Test' },
  { key: 'thresholdMethod', label: 'Threshold Method' },
  { key: 'thresholdValue', label: 'Threshold Value' },
  { key: 'driftScore', label: 'Drift Score' },
  { key: 'wmape', label: 'WMAPE' },
  { key: 'rmse', label: 'RMSE' },
  { key: 'mae', label: 'MAE' },
]

export default function UnderlyingMetricsPanel({ metrics }) {
  return (
    <ExpandablePanel
      title="Show underlying metrics"
      subtitle="Raw metric payload the narrative and decision were generated from"
    >
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        {metricFields.map((field) => (
          <div key={field.key}>
            <p className="text-xs text-slate-400">{field.label}</p>
            <p className="mt-1 text-sm font-semibold text-slate-700 dark:text-slate-200">
              {metrics[field.key]}
            </p>
          </div>
        ))}
        <div>
          <p className="text-xs text-slate-400">Validation Result</p>
          <div className="mt-1.5">
            <Badge status={metrics.validationResult}>{metrics.validationResult}</Badge>
          </div>
        </div>
      </div>
    </ExpandablePanel>
  )
}
