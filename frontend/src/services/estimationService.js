import { apiClient } from './apiClient'

/**
 * Pre-run duration and cost estimate for a configuration.
 *
 * Sends exactly what a run would be submitted with, so the estimate
 * describes the run the user is about to start rather than an
 * approximation of it. Runs no forecasting code — see the backend's
 * EstimationService.
 */
export function estimateRun({ fileId, mapping, selectedModels, horizon, aggregationMethod }) {
  const metadata = {
    date_column: mapping.dateColumn,
    target_column: mapping.targetColumn,
    key_columns: mapping.keyColumns,
    feature_columns: mapping.featureColumns,
  }
  if (aggregationMethod) {
    metadata.aggregation_method = aggregationMethod
  }

  return apiClient.post('/estimate', {
    file_id: fileId,
    metadata,
    selected_models: selectedModels,
    horizon,
  })
}
