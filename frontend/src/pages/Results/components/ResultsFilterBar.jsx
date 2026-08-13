import Select from '../../../components/ui/Select'

function FilterField({ label, children }) {
  return (
    <div className="min-w-[180px] flex-1">
      <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide text-slate-400">
        {label}
      </label>
      {children}
    </div>
  )
}

// Renders the filter row only — no Card wrapper — so it can be embedded
// directly inside the chart section it controls. Every option list comes
// from the loaded run, never from static data.
//
// Dataset and Run are dependent: `runOptions` is expected to already be
// filtered down to the selected dataset's own runs (newest first) — this
// component just renders whatever it is given, never re-derives it.
export default function ResultsFilterBar({
  dataset,
  datasetOptions = [],
  onDatasetChange,
  run,
  runOptions = [],
  onRunChange,
  businessKey,
  businessKeyOptions = [],
  onBusinessKeyChange,
  horizon,
  horizonOptions = [],
  onHorizonChange,
}) {
  return (
    <div className="flex flex-col gap-4 sm:flex-row sm:items-end">
      <FilterField label="Dataset">
        <Select value={dataset} onChange={onDatasetChange} options={datasetOptions} placeholder="Select dataset" />
      </FilterField>
      <FilterField label="Run">
        <Select value={run} onChange={onRunChange} options={runOptions} placeholder="Select run" />
      </FilterField>
      <FilterField label="Business Key">
        <Select
          value={businessKey}
          onChange={onBusinessKeyChange}
          options={businessKeyOptions}
          placeholder="Select key"
        />
      </FilterField>
      <FilterField label="Forecast Horizon">
        <Select
          value={horizon}
          onChange={onHorizonChange}
          options={horizonOptions}
          placeholder="Select horizon"
        />
      </FilterField>
    </div>
  )
}
