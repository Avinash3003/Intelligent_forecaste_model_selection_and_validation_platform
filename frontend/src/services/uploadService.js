import { apiClient } from './apiClient'

// Uploads a dataset file and returns { success, file_id, filename, size_bytes, message }.
export function uploadDataset(file, { onProgress } = {}) {
  const formData = new FormData()
  formData.append('file', file)
  return apiClient.postForm('/upload', formData, { onProgress })
}

// Profiles a previously uploaded dataset (basic, metadata-free inspection).
// Returns { dataset_name, total_rows, total_columns, columns: [...] }.
export function profileDataset(fileId) {
  return apiClient.post('/profile', { file_id: fileId })
}

// The dataset's real date coverage for whichever column the user has
// tentatively assigned as the date column in Metadata Mapping — the same
// concept and field names as the Results page's Data Coverage (Priority
// #8), just sourced from the upload instead of a completed run's summary.
// Returns { available, date_range_start, date_range_end }.
export function fetchDatasetDateRange(fileId, dateColumn) {
  return apiClient.post('/profile/date-range', { file_id: fileId, date_column: dateColumn })
}
