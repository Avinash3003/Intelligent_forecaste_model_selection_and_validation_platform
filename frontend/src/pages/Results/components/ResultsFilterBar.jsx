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
export default function ResultsFilterBar({
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
