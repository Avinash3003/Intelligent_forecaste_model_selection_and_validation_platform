import Select from '../ui/Select'

/**
 * The Dataset -> Run selector row, shared by Experiments and Observability
 * (Results embeds the same two controls inside its own chart filter bar,
 * alongside Business Key and Horizon).
 *
 * Presentational only: it renders whatever options it is handed and never
 * derives them — `useDatasetRunFilter` owns that logic for every caller.
 */
export default function DatasetRunFilter({
  dataset,
  datasetOptions = [],
  onDatasetChange,
  run,
  runOptions = [],
  onRunChange,
  className = '',
}) {
  return (
    <div className={`grid grid-cols-1 gap-4 sm:grid-cols-2 ${className}`}>
      <div className="min-w-0">
        <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide text-slate-400">
          Dataset
        </label>
        <Select
          value={dataset}
          onChange={onDatasetChange}
          options={datasetOptions}
          placeholder="Select dataset"
        />
      </div>
      <div className="min-w-0">
        <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide text-slate-400">
          Run
        </label>
        <Select value={run} onChange={onRunChange} options={runOptions} placeholder="Select run" />
      </div>
    </div>
  )
}
