/**
 * The backend's `Permission` vocabulary, verbatim
 * (backend/app/auth/models.py).
 *
 * Mirrored rather than derived so a typo is a broken import instead of a
 * silently-always-false check. The role -> permission *mapping* lives only
 * on the server; the client is told which permissions it holds and never
 * recomputes them.
 */
export const PERMISSIONS = {
  DATASET_UPLOAD: 'dataset:upload',
  DATASET_READ: 'dataset:read',
  FORECAST_CONFIGURE: 'forecast:configure',
  FORECAST_ESTIMATE: 'forecast:estimate',
  FORECAST_RUN: 'forecast:run',
  RUN_READ: 'run:read',
  RUN_CANCEL: 'run:cancel',
  RESULTS_READ: 'results:read',
  MODEL_INSPECT: 'model:inspect',
  ADMIN_MANAGE: 'admin:manage',
}
