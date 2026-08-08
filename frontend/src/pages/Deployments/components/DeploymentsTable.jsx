import { useNavigate } from 'react-router-dom'
import { Eye } from 'lucide-react'
import DataTable from '../../../components/tables/DataTable'
import Badge from '../../../components/ui/Badge'
import ProgressBar from '../../../components/ui/ProgressBar'

export default function DeploymentsTable({ runs }) {
  const navigate = useNavigate()

  const columns = [
    {
      key: 'id',
      header: 'Run',
      render: (r) => (
        <div>
          <p className="font-medium text-slate-700 dark:text-slate-200">{r.displayName}</p>
          <p className="font-mono text-xs text-slate-400">{r.id}</p>
        </div>
      ),
    },
    { key: 'dataset', header: 'Dataset' },
    { key: 'status', header: 'Status', render: (r) => <Badge status={r.status}>{r.status}</Badge> },
    { key: 'startTime', header: 'Start Time (IST)' },
    { key: 'duration', header: 'Duration' },
    {
      key: 'progress',
      header: 'Progress',
      render: (r) => (
        <div className="min-w-[140px]">
          <ProgressBar value={r.progress} showLabel />
        </div>
      ),
    },
    {
      key: 'action',
      header: 'Action',
      render: (r) => (
        <button
          onClick={() => navigate(`/deployments/${r.id}`)}
          className="flex items-center gap-1.5 text-sm font-medium text-brand-600 hover:text-brand-700 dark:text-brand-400"
        >
          <Eye size={14} /> View Details
        </button>
      ),
    },
  ]

  return <DataTable columns={columns} data={runs} />
}
