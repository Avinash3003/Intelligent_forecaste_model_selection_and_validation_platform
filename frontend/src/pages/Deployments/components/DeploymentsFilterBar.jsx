import SearchBox from '../../../components/ui/SearchBox'
import Select from '../../../components/ui/Select'
import { JOB_STATUSES } from '../../../services'

// Built from the backend's own JobStatus values so a filter option can
// never reference a status the API does not produce.
const statusFilters = ['All', ...JOB_STATUSES]

export default function DeploymentsFilterBar({ search, onSearchChange, status, onStatusChange }) {
  return (
    <div className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-center">
      <SearchBox
        value={search}
        onChange={(e) => onSearchChange(e.target.value)}
        placeholder="Search by run ID or dataset..."
        className="sm:max-w-xs"
      />
      <div className="sm:w-48">
        <Select
          value={status}
          onChange={onStatusChange}
          options={statusFilters}
          placeholder="Filter by status"
        />
      </div>
    </div>
  )
}
