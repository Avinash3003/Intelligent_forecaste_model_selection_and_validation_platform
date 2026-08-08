import { Routes, Route } from 'react-router-dom'
import MainLayout from '../layouts/MainLayout'
import Dashboard from '../pages/Dashboard/Dashboard'
import ForecastPipeline from '../pages/ForecastPipeline/ForecastPipeline'
import Deployments from '../pages/Deployments/Deployments'
import PipelineDetails from '../pages/Deployments/PipelineDetails'
import Results from '../pages/Results/Results'
import MLflowExperiments from '../pages/MLflowExperiments/MLflowExperiments'
import Settings from '../pages/Settings/Settings'

export default function AppRoutes() {
  return (
    <Routes>
      <Route element={<MainLayout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/forecast-pipeline" element={<ForecastPipeline />} />
        <Route path="/deployments" element={<Deployments />} />
        <Route path="/deployments/:runId" element={<PipelineDetails />} />
        <Route path="/results" element={<Results />} />
        <Route path="/mlflow-experiments" element={<MLflowExperiments />} />
        <Route path="/settings" element={<Settings />} />
      </Route>
    </Routes>
  )
}
