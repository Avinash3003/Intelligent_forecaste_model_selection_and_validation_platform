import { AlertCircle, CheckCircle2, AlertTriangle, XCircle, Loader2, ShieldCheck, CalendarClock } from 'lucide-react'
import SectionContainer from '../../../components/layout/SectionContainer'
import Select from '../../../components/ui/Select'
import MultiSelect from '../../../components/ui/MultiSelect'
import Checkbox from '../../../components/ui/Checkbox'
import Badge from '../../../components/ui/Badge'
import Button from '../../../components/ui/Button'
import Loader from '../../../components/ui/Loader'
import { aggregationMethodOptions, derivedFeatureOptions } from '../../../data/appConfig'

function Field({ label, required, hint, error, children }) {
  return (
    <div>
      <label className="mb-1.5 flex items-center gap-1.5 text-sm font-medium text-slate-600 dark:text-slate-300">
        {label}
        {required && <span className="text-rose-500">*</span>}
      </label>
      {children}
      {hint && !error && <p className="mt-1.5 text-xs text-slate-400">{hint}</p>}
      {error && <p className="mt-1.5 text-xs text-rose-500">{error}</p>}
    </div>
  )
}

// Icon per validation status. The backend owns the verdict; this only
// chooses how to render it.
const STATUS_ICON = {
  Valid: { Icon: CheckCircle2, className: 'text-emerald-600 dark:text-emerald-400' },
  Convertible: { Icon: AlertTriangle, className: 'text-amber-600 dark:text-amber-400' },
  Invalid: { Icon: XCircle, className: 'text-rose-600 dark:text-rose-400' },
}

const SUITABILITY_TONE = {
  Ready: {
    Icon: CheckCircle2,
    container: 'border-emerald-200 bg-emerald-50/70 dark:border-emerald-800 dark:bg-emerald-900/20',
    icon: 'text-emerald-600 dark:text-emerald-400',
    text: 'text-emerald-700 dark:text-emerald-300',
  },
  Warnings: {
    Icon: AlertTriangle,
    container: 'border-amber-200 bg-amber-50/70 dark:border-amber-800 dark:bg-amber-900/20',
    icon: 'text-amber-600 dark:text-amber-400',
    text: 'text-amber-700 dark:text-amber-300',
  },
  'Not Suitable': {
    Icon: XCircle,
    container: 'border-rose-200 bg-rose-50/70 dark:border-rose-800 dark:bg-rose-900/20',
    icon: 'text-rose-600 dark:text-rose-400',
    text: 'text-rose-700 dark:text-rose-300',
  },
}

function ValidationRow({ check }) {
  const { Icon, className } = STATUS_ICON[check.status] ?? STATUS_ICON.Valid

  return (
    <li className="flex flex-col gap-3 border-b border-slate-100 py-4 first:pt-0 last:border-0 last:pb-0 dark:border-slate-800/60 sm:flex-row sm:items-start sm:justify-between">
      <div className="flex items-start gap-3">
        <Icon size={18} className={`mt-0.5 shrink-0 ${className}`} />
        <div>
          <p className="text-sm font-semibold text-slate-700 dark:text-slate-200">{check.title}</p>
          <p className="mt-0.5 text-sm text-slate-400">{check.description}</p>
        </div>
      </div>
      <Badge status={check.status} className="shrink-0 self-start">
        {check.statusLabel || check.status}
      </Badge>
    </li>
  )
}

export default function StepMetadataMapping({
  mapping,
  onChange,
  errors,
  columns,
  validating,
  validationResult,
  validationError,
  onValidate,
  aggregationRequired,
  aggregationMethod,
  onAggregationMethodChange,
}) {
  // Every dropdown lists all remaining columns — the frontend enforces only
  // role-exclusivity (a column cannot hold two roles). Whether a chosen
  // column is actually usable as a date/target/key is decided by the
  // backend Validation Engine, never here.
  const toOption = (c) => ({ value: c.name, label: c.name, sublabel: c.dtype })

  const dateColumnOptions = columns.map(toOption)
  const targetColumnOptions = columns.filter((c) => c.name !== mapping.dateColumn).map(toOption)
  const keyColumnOptions = columns
    .filter((c) => c.name !== mapping.dateColumn && c.name !== mapping.targetColumn)
    .map(toOption)
  const featureColumnOptions = keyColumnOptions.filter((opt) => !mapping.keyColumns.includes(opt.value))

  const suitability = validationResult?.forecastSuitability
  const suitabilityTone = suitability ? SUITABILITY_TONE[suitability.status] : null

  // Aggregation only applies to grains finer than monthly. Monthly data
  // needs no roll-up, and coarser grains are rejected by the backend's
  // frequency check — the reason is already shown in the report above, so
  // this section stays hidden in both cases. The verdict comes from the
  // backend so this rule lives in exactly one place.
  const detectedFrequency = validationResult?.configurationSummary?.forecastFrequency

  // "Validated" means the backend judged the mapping deployable — Ready or
  // Warnings (convertible columns). Not Suitable leaves the user on this step.
  const isValidated = Boolean(validationResult?.readyForDeployment)

  return (
    <div className="space-y-5">
      <SectionContainer
        title="Metadata mapping"
        subtitle="Assign a role to each column — the platform validates your selections against the dataset"
      >
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
          <Field label="Date Column" required error={errors?.dateColumn}>
            <Select
              value={mapping.dateColumn}
              onChange={(v) => onChange('dateColumn', v)}
              options={dateColumnOptions}
              placeholder="Select date column"
              error={!!errors?.dateColumn}
            />
          </Field>

          <Field label="Target Column" required error={errors?.targetColumn}>
            <Select
              value={mapping.targetColumn}
              onChange={(v) => onChange('targetColumn', v)}
              options={targetColumnOptions}
              placeholder="Select target column"
              error={!!errors?.targetColumn}
            />
          </Field>

          <Field label="Key Column(s)" hint="Leave empty to forecast the dataset as a single series.">
            <MultiSelect
              values={mapping.keyColumns}
              onChange={(v) => onChange('keyColumns', v)}
              options={keyColumnOptions}
              placeholder="Select business key column(s)"
            />
          </Field>

          <Field label="Feature Column(s)" hint="Optional exogenous regressors.">
            <MultiSelect
              values={mapping.featureColumns}
              onChange={(v) => onChange('featureColumns', v)}
              options={featureColumnOptions}
              placeholder="Select feature column(s)"
            />
          </Field>
        </div>

        {/* Validation is explicit: the user validates here and reads the
            report in place. Editing any selection clears the result, so the
            report on screen always matches the current mapping. */}
        <div className="mt-6 flex flex-col gap-3 border-t border-slate-100 pt-5 dark:border-slate-800 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-sm text-slate-400">
            {isValidated
              ? 'Metadata validated — you can continue to forecast configuration.'
              : 'Validate your metadata selections before continuing.'}
          </p>
          <Button onClick={onValidate} disabled={validating} className="shrink-0">
            {validating ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <ShieldCheck size={16} />
            )}
            {validating ? 'Validating…' : isValidated ? 'Re-validate metadata' : 'Validate metadata'}
          </Button>
        </div>
      </SectionContainer>

      {/* Distinct from "Feature Column(s)" above (existing raw exogenous
          columns, passed through unchanged): these are engineered from the
          target/date columns themselves — lag and rolling-mean values of
          past target values, plus calendar position — for XGBoost/LightGBM
          only. Every other model has its own native trend/seasonality
          handling and ignores this selection entirely. */}
      <SectionContainer
        title="Feature Columns"
        subtitle="Derived columns (lag, rolling mean, calendar) included in the curated dataset for XGBoost/LightGBM — controls what those models actually train on"
      >
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {derivedFeatureOptions.map((option) => (
            <Checkbox
              key={option.id}
              label={option.label}
              checked={mapping.derivedFeatures.includes(option.id)}
              onChange={(checked) =>
                onChange(
                  'derivedFeatures',
                  checked
                    ? [...mapping.derivedFeatures, option.id]
                    : mapping.derivedFeatures.filter((id) => id !== option.id)
                )
              }
            />
          ))}
        </div>
      </SectionContainer>

      {validating && (
        <SectionContainer title="Metadata validation report">
          <Loader label="Validating metadata against the uploaded dataset…" />
        </SectionContainer>
      )}

      {!validating && validationError && (
        <SectionContainer title="Metadata validation report">
          <div className="flex items-start gap-2.5 rounded-xl border border-rose-200 bg-rose-50/70 px-4 py-3 text-sm text-rose-600 dark:border-rose-800 dark:bg-rose-900/20 dark:text-rose-300">
            <AlertCircle size={16} className="mt-0.5 shrink-0" />
            <span>{validationError}</span>
          </div>
        </SectionContainer>
      )}

      {!validating && validationResult && (
        <SectionContainer
          title="Metadata validation report"
          subtitle="Each selection validated against the dataset by the platform"
        >
          <ul>
            {validationResult.checks.map((check) => (
              <ValidationRow key={check.id} check={check} />
            ))}
          </ul>
        </SectionContainer>
      )}

      {!validating && aggregationRequired && (
        <SectionContainer
          title="Monthly aggregation"
          subtitle={`This dataset is detected as ${detectedFrequency}. The forecasting pipeline requires monthly data.`}
        >
          <div className="flex items-start gap-3 rounded-xl border border-brand-200 bg-brand-50/60 px-4 py-3.5 dark:border-brand-800 dark:bg-brand-900/20">
            <CalendarClock size={18} className="mt-0.5 shrink-0 text-brand-600 dark:text-brand-400" />
            <p className="text-sm text-slate-600 dark:text-slate-300">
              Select how the target column should be aggregated while converting to monthly
              frequency. The conversion is applied during preprocessing after deployment.
            </p>
          </div>

          <div className="mt-5 max-w-md">
            <label className="mb-1.5 flex items-center gap-1.5 text-sm font-medium text-slate-600 dark:text-slate-300">
              Aggregation Method
              <span className="text-rose-500">*</span>
            </label>
            <Select
              value={aggregationMethod}
              onChange={onAggregationMethodChange}
              options={aggregationMethodOptions}
              placeholder="Select aggregation method"
            />
          </div>
        </SectionContainer>
      )}

      {!validating && suitability && suitabilityTone && (
        <SectionContainer title="Forecast suitability">
          <div className={`rounded-xl border px-4 py-4 ${suitabilityTone.container}`}>
            <div className="flex items-start gap-3">
              <suitabilityTone.Icon size={20} className={`mt-0.5 shrink-0 ${suitabilityTone.icon}`} />
              <div className="flex-1">
                <div className="flex flex-wrap items-center gap-3">
                  <p className={`text-sm font-semibold ${suitabilityTone.text}`}>{suitability.summary}</p>
                  <Badge status={suitability.status}>{suitability.status}</Badge>
                </div>

                {suitability.reasons.length > 0 && (
                  <ul className="mt-3 list-inside list-disc space-y-1">
                    {suitability.reasons.map((reason, idx) => (
                      <li key={idx} className={`text-sm ${suitabilityTone.text}`}>
                        {reason}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          </div>
        </SectionContainer>
      )}
    </div>
  )
}
