import { apiClient } from './apiClient'

// Uploads a dataset file and returns { success, file_id, filename, size_bytes, message }.
export function uploadDataset(file) {
  const formData = new FormData()
  formData.append('file', file)
  return apiClient.postForm('/upload', formData)
}

// Profiles a previously uploaded dataset (basic, metadata-free inspection).
// Returns { dataset_name, total_rows, total_columns, columns: [...] }.
export function profileDataset(fileId) {
  return apiClient.post('/profile', { file_id: fileId })
}
