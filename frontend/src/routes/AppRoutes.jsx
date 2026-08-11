import { Routes, Route } from 'react-router-dom'
import MainLayout from '../layouts/MainLayout'
import Dashboard from '../pages/Dashboard/Dashboard'
import ForecastPipeline from '../pages/ForecastPipeline/ForecastPipeline'
import Deployments from '../pages/Deployments/Deployments'
import PipelineDetails from '../pages/Deployments/PipelineDetails'
import Results from '../pages/Results/Results'
import MLflowExperiments from '../pages/MLflowExperiments/MLflowExperiments'
import LLMOps from '../pages/LLMOps/LLMOps'
import Settings from '../pages/Settings/Settings'
import RequirePermission from '../auth/RequirePermission'
import { PERMISSIONS } from '../auth/permissions'

// Each route names the permission it needs. The sidebar hides links a user
// cannot use, so this guard is mostly for direct navigation — and the API
// enforces the same rule again on every request behind these pages.
export default function AppRoutes() {
  return (
    <Routes>
      <Route element={<MainLayout />}>
        <Route path="/" element={<Dashboard />} />
        <Route
          path="/forecast-pipeline"
          element={
            <RequirePermission permission={PERMISSIONS.FORECAST_RUN}>
              <ForecastPipeline />
            </RequirePermission>
          }
        />
        <Route
          path="/deployments"
          element={
            <RequirePermission permission={PERMISSIONS.RUN_READ}>
              <Deployments />
            </RequirePermission>
          }
        />
        <Route
          path="/deployments/:runId"
          element={
            <RequirePermission permission={PERMISSIONS.RUN_READ}>
              <PipelineDetails />
            </RequirePermission>
          }
        />
        <Route
          path="/results"
          element={
            <RequirePermission permission={PERMISSIONS.RESULTS_READ}>
              <Results />
            </RequirePermission>
          }
        />
        <Route
          path="/mlflow-experiments"
          element={
            <RequirePermission permission={PERMISSIONS.MODEL_INSPECT}>
              <MLflowExperiments />
            </RequirePermission>
          }
        />
        <Route
          path="/llmops-observability"
          element={
            <RequirePermission permission={PERMISSIONS.MODEL_INSPECT}>
              <LLMOps />
            </RequirePermission>
          }
        />
        <Route path="/settings" element={<Settings />} />
      </Route>
    </Routes>
  )
}
