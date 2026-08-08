import { apiClient } from './apiClient'

// Validates a Date/Target/Key/Feature column mapping against the uploaded
// dataset. Returns { mode, normalized_config, checks, ready_for_deployment,
// configuration_summary }.
export function validateMetadata({ fileId, dateColumn, targetColumn, keyColumns, featureColumns }) {
  return apiClient.post('/metadata/validate', {
    file_id: fileId,
    date_column: dateColumn,
    target_column: targetColumn,
    key_columns: keyColumns,
    feature_columns: featureColumns,
  })
}
