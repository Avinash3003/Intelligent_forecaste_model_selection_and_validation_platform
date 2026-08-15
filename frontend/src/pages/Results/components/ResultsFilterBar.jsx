import Select from '../../../components/ui/Select'

function FilterField({ label, children }) {
  // `min-w-0` is what actually prevents the overflow: a grid/flex child
  // defaults to min-width:auto, so a long option label (a run id, a
  // multi-part business key) forces the track wider than its share and
  // pushes the last field off the card. Truncation happens inside the
  // control instead.
  return (
    <div className="min-w-0">
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
  // A 2-up grid at card width, 4-up only once there is genuinely room for
  // it. The previous single flex row asked for 4 x 180px minimum, which the
  // chart card cannot give beside the AI Insight panel — so Forecast
  // Horizon was pushed past the card's right edge.
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4 xl:items-end">
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
