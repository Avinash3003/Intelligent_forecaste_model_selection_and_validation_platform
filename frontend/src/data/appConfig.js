// Static application configuration.
//
// Everything here is a fixed catalog or option list that the platform
// genuinely defines client-side — navigation, the model registry's display
// names, supported upload formats, and the forecast-horizon scale. No run,
// metric, cost or status data lives here: all of that comes from the API.

export const currentUser = {
  name: 'Avinash Reddy',
  email: 'avinashreddy5142@gmail.com',
  role: 'Platform Admin',
  initials: 'AR',
  avatarColor: 'bg-brand-600',
}

export const projectInfo = {
  name: 'Forecast IQ',
  tagline: 'Intelligent Forecast Model Selection & Validation Platform',
}

export const sidebarNav = [
  { label: 'Dashboard', path: '/', icon: 'LayoutDashboard' },
  { label: 'Forecast Pipeline', path: '/forecast-pipeline', icon: 'GitBranch' },
  { label: 'Deployments', path: '/deployments', icon: 'Rocket' },
  { label: 'Results', path: '/results', icon: 'LineChart' },
  { label: 'MLflow Experiments', path: '/mlflow-experiments', icon: 'FlaskConical' },
  { label: 'Settings', path: '/settings', icon: 'Settings' },
]

// ---------------------------------------------------------------------------
// Forecast Pipeline wizard
// ---------------------------------------------------------------------------

export const forecastPipelineSteps = [
  { id: 1, key: 'upload', label: 'Dataset Upload' },
  { id: 2, key: 'profiling', label: 'Dataset Profiling' },
  { id: 3, key: 'mapping', label: 'Metadata Mapping' },
  { id: 4, key: 'configuration', label: 'Forecast Configuration' },
  { id: 5, key: 'review', label: 'Review & Deploy' },
]

export const supportedFileFormats = [
  { format: 'CSV', extension: '.csv' },
  { format: 'Excel', extension: '.xlsx' },
  { format: 'Excel (legacy)', extension: '.xls' },
]

// All forecasting runs at month level, so a Daily/Weekly dataset must be
// rolled up first. The right method depends on what the target measures,
// which only the user knows — hence an explicit choice rather than a
// default the platform guesses.
export const aggregationMethodOptions = [
  { value: 'sum', label: 'Sum', sublabel: 'Total across the month — for flow metrics like units sold' },
  { value: 'mean', label: 'Mean', sublabel: 'Average across the month — for rates and levels like price' },
  { value: 'last', label: 'Last', sublabel: 'Final value in the month — for point-in-time levels like inventory' },
]

export const defaultAggregationMethod = 'sum'

// Fallback pre-selected when the wizard opens. Must be one of the ids in
// `forecastModels` — the fallback always participates in evaluation too.
export const defaultFallbackModel = 'lightgbm'

export const forecastModels = [
  {
    id: 'prophet',
    name: 'Prophet',
    description: 'Strong for stable seasonal series with limited exogenous drivers.',
  },
  {
    id: 'arima',
    name: 'ARIMA / SARIMA',
    description: 'Good for low-volatility series with clear autocorrelation.',
  },
  {
    id: 'lightgbm',
    name: 'LightGBM',
    description: 'Fast, scales well across thousands of keys.',
  },
  {
    id: 'xgboost',
    name: 'XGBoost',
    description: 'Handles nonlinear feature interactions and regressors well.',
  },
  {
    id: 'tft',
    name: 'TFT',
    description: 'Deep model, best with long history and rich regressors.',
  },
]

// Forecast horizon is chosen on a continuous month scale (6 months–5 years)
// rather than from fixed options, so a user can pick any exact horizon. Year
// marks are supplied as tick labels purely to orient the scale.
export const forecastHorizonRange = { min: 6, max: 60, step: 1 }

export const forecastHorizonTicks = [
  { value: 6, label: '6m' },
  { value: 12, label: '1y' },
  { value: 24, label: '2y' },
  { value: 36, label: '3y' },
  { value: 48, label: '4y' },
  { value: 60, label: '5y' },
]

export const defaultForecastHorizon = 12
