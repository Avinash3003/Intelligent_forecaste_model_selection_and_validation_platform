import * as Icons from 'lucide-react'
import { AlertCircle } from 'lucide-react'
import Card from '../../../components/ui/Card'
import Button from '../../../components/ui/Button'
import SectionContainer from '../../../components/layout/SectionContainer'
import DataTable from '../../../components/tables/DataTable'
import Loader from '../../../components/ui/Loader'

function InspectionStat({ label, value, icon }) {
  const Icon = Icons[icon] || Icons.Activity
  return (
    <Card hover className="p-5">
      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand-50 text-brand-600 dark:bg-brand-900/30 dark:text-brand-400">
        <Icon size={19} strokeWidth={2} />
      </div>
      <p className="mt-4 text-xl font-bold tracking-tight text-slate-800 dark:text-slate-100">{value}</p>
      <p className="text-sm text-slate-400 mt-0.5">{label}</p>
    </Card>
  )
}

// Reports the raw pandas dtype exactly as read from the file. No
// forecasting judgement is made here — whether an "object" column can
// serve as a date or target is decided by the backend Validation Engine
// once the user has assigned roles.
const schemaColumns = [
  { key: 'name', header: 'Column Name' },
  { key: 'dtype', header: 'Data Type' },
  { key: 'sampleValue', header: 'Sample Value' },
  { key: 'nullPct', header: 'Null %' },
  { key: 'distinctValues', header: 'Distinct Values' },
]

export default function StepDatasetProfiling({ inspection, loading, error, onRetry }) {
  if (loading) {
    return (
      <SectionContainer title="Basic dataset inspection">
        <Loader label="Profiling uploaded dataset…" />
      </SectionContainer>
    )
  }

  if (error) {
    return (
      <SectionContainer title="Basic dataset inspection">
        <div className="flex flex-col items-center gap-4 py-14 text-center">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-rose-50 text-rose-600 dark:bg-rose-900/30 dark:text-rose-400">
            <AlertCircle size={22} />
          </div>
          <p className="max-w-sm text-sm text-slate-500 dark:text-slate-400">{error}</p>
          <Button variant="secondary" size="sm" onClick={onRetry}>
            Retry
          </Button>
        </div>
      </SectionContainer>
    )
  }

  if (!inspection) return null

  return (
    <div className="space-y-5">
      <SectionContainer
        title="Basic dataset inspection"
        subtitle="Information read directly from the uploaded file — no metadata has been mapped yet"
      >
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <InspectionStat label="Dataset Name" value={inspection.datasetName} icon="FileText" />
          <InspectionStat label="Total Rows" value={inspection.totalRows} icon="Rows3" />
          <InspectionStat label="Total Columns" value={inspection.totalColumns} icon="Columns3" />
          <InspectionStat label="File Size" value={inspection.fileSize} icon="HardDrive" />
        </div>
      </SectionContainer>

      <SectionContainer
        title="Column profile"
        subtitle="Raw data types as read from the file, with sample values and completeness"
      >
        <DataTable columns={schemaColumns} data={inspection.columns.map((c) => ({ id: c.name, ...c }))} />
      </SectionContainer>
    </div>
  )
}
