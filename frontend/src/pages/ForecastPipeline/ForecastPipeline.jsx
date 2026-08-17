import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import PageContainer from '../../components/common/PageContainer'
import StepIndicator from '../../components/common/StepIndicator'
import Card from '../../components/ui/Card'
import StepDatasetUpload from './components/StepDatasetUpload'
import StepDatasetProfiling from './components/StepDatasetProfiling'
import StepMetadataMapping from './components/StepMetadataMapping'
import StepForecastConfiguration from './components/StepForecastConfiguration'
import StepReviewDeploy from './components/StepReviewDeploy'
import WizardFooter from './components/WizardFooter'
import { uploadDataset, profileDataset, fetchDatasetDateRange, validateMetadata, deployRun, estimateRun } from '../../services'
import {
  forecastPipelineSteps,
  forecastModels,
  defaultFallbackModel,
  defaultForecastHorizon,
  defaultAggregationMethod,
  defaultDerivedFeatures,
} from '../../data/appConfig'

const initialMapping = {
  dateColumn: '',
  targetColumn: '',
  keyColumns: [],
  featureColumns: [],
  // Derived feature columns for XGBoost/LightGBM (Priority C) — defaults
  // to every supported feature, reproducing pre-existing behavior exactly
  // until a user actually unchecks something.
  derivedFeatures: defaultDerivedFeatures,
}

const initialConfig = {
  horizon: defaultForecastHorizon,
  selectedModels: forecastModels.map((m) => m.id),
  fallbackModel: defaultFallbackModel,
}

// Maps the backend's snake_case /profile response into the shape the
// wizard's step components consume. `dtype` is passed through as the raw
// pandas type — profiling makes no forecasting judgement.
function normalizeProfileResponse(response) {
  return {
    datasetName: response.dataset_name,
    totalRows: response.total_rows.toLocaleString(),
    totalColumns: response.total_columns,
    fileSize: response.file_size,
    columns: response.columns.map((column) => ({
      name: column.name,
      dtype: column.dtype,
      sampleValue: column.sample_value,
      nullPct: column.null_pct,
      distinctValues: column.distinct_values,
    })),
  }
}

// Maps the backend's snake_case /metadata/validate response into the shape
// StepMetadataMapping renders. Every verdict here originates from the
// backend Validation Engine — the frontend re-derives nothing.
function normalizeValidationResponse(response) {
  return {
    mode: response.mode,
    checks: response.checks.map((check) => ({
      id: check.id,
      title: check.title,
      status: check.status,
      statusLabel: check.status_label,
      description: check.description,
    })),
    forecastSuitability: {
      status: response.forecast_suitability.status,
      summary: response.forecast_suitability.summary,
      reasons: response.forecast_suitability.reasons,
    },
    readyForDeployment: response.ready_for_deployment,
    configurationSummary: {
      datasetName: response.configuration_summary.dataset_name,
      rows: response.configuration_summary.rows,
      columns: response.configuration_summary.columns,
      forecastMode: response.configuration_summary.forecast_mode,
      forecastFrequency: response.configuration_summary.forecast_frequency,
      uniqueBusinessKeys: response.configuration_summary.unique_business_keys,
      aggregationRequired: response.configuration_summary.aggregation_required,
    },
  }
}

export default function ForecastPipeline() {
  const [currentStep, setCurrentStep] = useState(1)
  const [maxReachedStep, setMaxReachedStep] = useState(1)
  const [errors, setErrors] = useState({})

  // Step 1 — Upload
  const [file, setFile] = useState(null)
  const [fileId, setFileId] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState(null)

  // Step 2 — Basic Dataset Inspection (auto-triggered after upload)
  const [inspection, setInspection] = useState(null)
  const [profiling, setProfiling] = useState(false)
  const [profileError, setProfileError] = useState(null)

  // Step 3 — Metadata Mapping + Validation
  const [mapping, setMapping] = useState(initialMapping)
  const [validating, setValidating] = useState(false)
  const [validationResult, setValidationResult] = useState(null)
  const [validationError, setValidationError] = useState(null)

  // Data Coverage (Priority B) — the real date range for whichever column
  // is currently assigned as the date column, shown on the Basic Dataset
  // Inspection card (Step 2). Unavailable until a date column exists, and
  // re-fetched every time it changes; never a stale value from a previous
  // dataset or a previous column choice.
  const [dateRange, setDateRange] = useState(null)

  // Chosen only when the detected grain is finer than monthly; carried into
  // the run configuration so preprocessing knows how to roll the target up.
  const [aggregationMethod, setAggregationMethod] = useState(defaultAggregationMethod)

  // Step 4 — Forecast Configuration (client-side only in this phase)
  const [config, setConfig] = useState(initialConfig)

  // Step 5 — Estimate & Run
  const [estimate, setEstimate] = useState(null)
  const [estimateLoading, setEstimateLoading] = useState(false)
  const [estimateError, setEstimateError] = useState(null)
  const [deploying, setDeploying] = useState(false)
  const [deployError, setDeployError] = useState(null)
  const [deployed, setDeployed] = useState(false)
  const [runId, setRunId] = useState(null)
  // Bumped on every loadEstimate() call so a response can tell whether it
  // is still the most recent request — going Next -> Previous -> edit
  // config -> Next again fires a second estimate before the first
  // resolves, and isBusy does not block that (estimateLoading is
  // deliberately excluded from it, since the estimate is advisory and
  // must never block navigation). Without this guard, whichever response
  // lands last wins regardless of which config it was computed for.
  const estimateRequestRef = useRef(0)

  const isBusy = uploading || profiling || validating || deploying

  // Whether the target must be rolled up to monthly. The backend owns this
  // rule (Monthly data needs no roll-up); the frontend never re-derives it.
  const aggregationRequired = Boolean(validationResult?.configurationSummary?.aggregationRequired)

  const loadProfile = async (id) => {
    setProfiling(true)
    setProfileError(null)
    try {
      const response = await profileDataset(id)
      setInspection(normalizeProfileResponse(response))
    } catch (err) {
      setProfileError(err.message)
    } finally {
      setProfiling(false)
    }
  }

  const handleFileSelect = async (selectedFile) => {
    if (uploading) return

    setFile(selectedFile)
    setUploadError(null)
    setInspection(null)
    setProfileError(null)
    setDateRange(null)
    setUploading(true)

    try {
      const response = await uploadDataset(selectedFile)
      setFileId(response.file_id)
      setErrors((prev) => ({ ...prev, file: undefined }))
      setUploading(false)
      await loadProfile(response.file_id)
    } catch (err) {
      setUploadError(err.message)
      setFile(null)
      setFileId(null)
      setUploading(false)
    }
  }

  const handleRemoveFile = () => {
    setFile(null)
    setFileId(null)
    setInspection(null)
    setUploadError(null)
    setProfileError(null)
    setDateRange(null)
    setErrors((prev) => ({ ...prev, file: undefined }))
  }

  // Re-fetch Data Coverage whenever the candidate date column changes (or
  // is cleared, or a new dataset replaces this one) — never left showing a
  // range computed for a different column or a different upload.
  useEffect(() => {
    if (!fileId || !mapping.dateColumn) {
      setDateRange(null)
      return
    }
    let cancelled = false
    fetchDatasetDateRange(fileId, mapping.dateColumn)
      .then((response) => {
        if (!cancelled) setDateRange(response)
      })
      .catch(() => {
        if (!cancelled) setDateRange(null)
      })
    return () => {
      cancelled = true
    }
  }, [fileId, mapping.dateColumn])

  const handleRetryProfile = () => {
    if (fileId) loadProfile(fileId)
  }

  const updateMapping = (field, value) => {
    setMapping((prev) => {
      const next = { ...prev, [field]: value }

      // Prevent a column from holding more than one role at a time.
      if (field === 'dateColumn' || field === 'targetColumn') {
        next.keyColumns = prev.keyColumns.filter((c) => c !== value)
        next.featureColumns = prev.featureColumns.filter((c) => c !== value)
      }
      if (field === 'keyColumns') {
        next.featureColumns = prev.featureColumns.filter((c) => !value.includes(c))
      }

      return next
    })
    setErrors((prev) => ({ ...prev, [field]: undefined }))
    // The mapping changed, so any previously fetched validation result is stale.
    setValidationResult(null)
    setValidationError(null)
  }

  const updateConfig = (field, value) => {
    setConfig((prev) => ({ ...prev, [field]: value }))
    setErrors((prev) => ({ ...prev, [field]: undefined }))
  }

  const goToStep = (step) => {
    if (isBusy) return
    if (step <= maxReachedStep) setCurrentStep(step)
  }

  const advanceTo = (step) => {
    setCurrentStep(step)
    setMaxReachedStep((prev) => Math.max(prev, step))
  }

  const handlePrevious = () => {
    if (isBusy) return
    if (currentStep > 1) setCurrentStep(currentStep - 1)
  }

  // Validation is an explicit, user-triggered action rather than a side
  // effect of pressing Next. The user stays on the mapping step to read the
  // report, and only advances once the backend has judged the mapping usable.
  const runValidation = async () => {
    if (validating) return

    const nextErrors = {}
    if (!mapping.dateColumn) nextErrors.dateColumn = 'Date column is required.'
    if (!mapping.targetColumn) nextErrors.targetColumn = 'Target column is required.'
    setErrors(nextErrors)
    if (Object.keys(nextErrors).length > 0) return

    setValidating(true)
    setValidationError(null)
    setValidationResult(null)
    try {
      const response = await validateMetadata({
        fileId,
        dateColumn: mapping.dateColumn,
        targetColumn: mapping.targetColumn,
        keyColumns: mapping.keyColumns,
        featureColumns: mapping.featureColumns,
      })
      setValidationResult(normalizeValidationResponse(response))
    } catch (err) {
      setValidationError(err.message)
    } finally {
      setValidating(false)
    }
  }

  // Duration/cost for the exact configuration about to be submitted.
  // A failure here never blocks the run — the estimate is advisory, and
  // refusing to forecast because we could not predict its cost would be
  // the wrong trade.
  const loadEstimate = async () => {
    if (!fileId) return
    const requestId = ++estimateRequestRef.current
    setEstimateLoading(true)
    setEstimateError(null)
    try {
      const response = await estimateRun({
        fileId,
        mapping,
        selectedModels: config.selectedModels,
        horizon: config.horizon,
        aggregationMethod: aggregationRequired ? aggregationMethod : null,
      })
      // A newer request has already started (Previous -> edit -> Next
      // again before this one resolved) — its own response, not this
      // stale one, owns the estimate/loading state from here.
      if (requestId !== estimateRequestRef.current) return
      setEstimate(response)
    } catch (err) {
      if (requestId !== estimateRequestRef.current) return
      setEstimateError(err.message)
    } finally {
      if (requestId === estimateRequestRef.current) setEstimateLoading(false)
    }
  }

  const handleNext = async () => {
    if (isBusy) return

    if (currentStep === 1) {
      if (!fileId) {
        setErrors({ file: 'Please upload a dataset to continue.' })
        return
      }
      setErrors({})
      advanceTo(2)
      return
    }

    if (currentStep === 2) {
      advanceTo(3)
      return
    }

    if (currentStep === 3) {
      // Next only navigates — the mapping must already have been validated
      // and judged deployable by the backend.
      if (!validationResult?.readyForDeployment) return
      setErrors({})
      advanceTo(4)
      return
    }

    if (currentStep === 4) {
      const configErrors = {}
      if (config.selectedModels.length === 0) {
        configErrors.selectedModels = 'Select at least one forecasting model to continue.'
      }
      if (!config.fallbackModel) {
        configErrors.fallbackModel = 'Select a default fallback model to continue.'
      }
      // No "fallback must be one of the selected models" rule: the fallback
      // is a baseline reached only after every candidate has failed
      // (Section 6.9), so constraining it to the candidate set would exclude
      // the simple, robust baselines that path exists for.

      setErrors(configErrors)
      if (Object.keys(configErrors).length > 0) return

      // The estimate is fetched on entry to step 5 rather than on a button
      // press: it is the reason that step exists, so making the user ask
      // for it would be an extra click before the only new information.
      loadEstimate()
      advanceTo(5)
      return
    }

    if (currentStep === 5) {
      setDeploying(true)
      setDeployError(null)
      try {
        const response = await deployRun({
          fileId,
          datasetName: file?.name,
          mapping,
          selectedModels: config.selectedModels,
          fallbackModel: config.fallbackModel,
          horizon: config.horizon,
          // Monthly data needs no roll-up, so no method is sent for it.
          aggregationMethod: aggregationRequired ? aggregationMethod : null,
        })
        setRunId(response.run_id)
        setDeployed(true)
      } catch (err) {
        setDeployError(err.message)
      } finally {
        setDeploying(false)
      }
    }
  }

  const nextLoading =
    currentStep === 1 ? uploading : currentStep === 2 ? profiling : currentStep === 5 ? deploying : false

  // Step 3 cannot be left until the backend has validated the mapping and
  // found it deployable. Any mapping edit clears the result, which disables
  // Next again until the user re-validates.
  // Aggregation must be chosen before leaving step 3 whenever the section
  // is shown; it always carries a default, so this only blocks if a user
  // explicitly clears it.
  const nextDisabled =
    currentStep === 3 &&
    (!validationResult?.readyForDeployment || (aggregationRequired && !aggregationMethod))

  return (
    <PageContainer>
      <div className="mb-6">
        <h1 className="text-2xl font-bold tracking-tight text-slate-800 dark:text-slate-100">
          Forecast Pipeline
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Upload, map columns, choose models, review the estimate, run.
        </p>
      </div>

      <Card className="mb-6 p-5">
        <StepIndicator
          steps={forecastPipelineSteps}
          currentStep={currentStep}
          maxReachedStep={deployed ? forecastPipelineSteps.length : maxReachedStep}
          onStepClick={goToStep}
        />
      </Card>

      <AnimatePresence mode="wait">
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.25, ease: 'easeOut' }}
        >
          {currentStep === 1 && (
            <StepDatasetUpload
              file={file}
              uploading={uploading}
              error={uploadError}
              onFileSelect={handleFileSelect}
              onRemove={handleRemoveFile}
            />
          )}
          {currentStep === 2 && (
            <StepDatasetProfiling
              inspection={inspection}
              loading={profiling}
              error={profileError}
              onRetry={handleRetryProfile}
              dateRange={dateRange}
            />
          )}
          {currentStep === 3 && (
            <StepMetadataMapping
              mapping={mapping}
              onChange={updateMapping}
              errors={errors}
              columns={inspection?.columns ?? []}
              validating={validating}
              validationResult={validationResult}
              validationError={validationError}
              onValidate={runValidation}
              aggregationRequired={aggregationRequired}
              aggregationMethod={aggregationMethod}
              onAggregationMethodChange={setAggregationMethod}
            />
          )}
          {currentStep === 4 && (
            <StepForecastConfiguration config={config} onChange={updateConfig} errors={errors} />
          )}
          {currentStep === 5 && (
            <StepReviewDeploy
              file={file}
              mapping={mapping}
              config={config}
              deployed={deployed}
              deployError={deployError}
              runId={runId}
              estimate={estimate}
              estimateLoading={estimateLoading}
              estimateError={estimateError}
            />
          )}
        </motion.div>
      </AnimatePresence>

      {errors.file && currentStep === 1 && (
        <p className="mt-3 text-sm text-rose-500">{errors.file}</p>
      )}

      <WizardFooter
        isFirstStep={currentStep === 1}
        isLastStep={currentStep === forecastPipelineSteps.length}
        deployed={deployed}
        loading={nextLoading}
        nextDisabled={nextDisabled}
        onPrevious={handlePrevious}
        onNext={handleNext}
      />
    </PageContainer>
  )
}
