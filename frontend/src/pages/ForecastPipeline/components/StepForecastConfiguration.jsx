import { useEffect } from 'react'
import { ShieldQuestion } from 'lucide-react'
import useModelAvailability from '../../../hooks/useModelAvailability'
import SectionContainer from '../../../components/layout/SectionContainer'
import RangeSlider from '../../../components/ui/RangeSlider'
import Checkbox from '../../../components/ui/Checkbox'
import Select from '../../../components/ui/Select'
import { formatMonths, formatMonthsShort } from '../../../utils/formatMonths'
import {
  forecastModels,
  fallbackModelOptions,
  forecastHorizonRange,
  forecastHorizonTicks,
} from '../../../data/appConfig'

export default function StepForecastConfiguration({ config, onChange, errors }) {
  // { modelId: reason } for models the configured execution mode cannot
  // run — empty when every model is runnable or the lookup failed.
  const unavailableModels = useModelAvailability()

  // The default selection is "every model", chosen before this component
  // knows which ones the environment supports. Once it does, anything
  // unrunnable is dropped: leaving it selected would submit a run that
  // silently reports the model Unavailable and would have the estimate
  // charge for work that never happens.
  useEffect(() => {
    const blocked = Object.keys(unavailableModels)
    if (!blocked.length) return
    const kept = config.selectedModels.filter((id) => !blocked.includes(id))
    if (kept.length !== config.selectedModels.length) {
      onChange('selectedModels', kept)
    }
  }, [unavailableModels, config.selectedModels, onChange])

  const toggleModel = (modelId, checked) => {
    const next = checked
      ? [...config.selectedModels, modelId]
      : config.selectedModels.filter((id) => id !== modelId)
    onChange('selectedModels', next)
  }

  // The fallback is a baseline, not a competitor — it is deliberately
  // independent of the evaluated set (Section 6.9). It only ever runs when
  // every candidate has already failed, so requiring it to also be one of
  // those candidates would rule out exactly the simple, robust baselines
  // the fallback path exists to reach for.
  const selectFallback = (modelId) => onChange('fallbackModel', modelId)

  const fallbackOptions = fallbackModelOptions.map((model) => ({
    value: model.id,
    label: model.name,
    sublabel: model.description,
  }))

  return (
    <div className="space-y-5">
      <SectionContainer
        title="Forecast horizon"
        subtitle="How far ahead to forecast — drag to any month between 6 months and 5 years"
      >
        <div className="max-w-2xl">
          <RangeSlider
            value={config.horizon}
            min={forecastHorizonRange.min}
            max={forecastHorizonRange.max}
            step={forecastHorizonRange.step}
            onChange={(value) => onChange('horizon', value)}
            ticks={forecastHorizonTicks}
            // Headline reads as a duration ("1 year 1 month"); the raw month
            // count stays visible beside it since that is the unit the
            // pipeline actually runs on.
            valueLabel={formatMonths(config.horizon)}
            hint={`${config.horizon} months`}
            formatBound={formatMonthsShort}
          />
        </div>
      </SectionContainer>

      <SectionContainer
        title="Forecast models"
        subtitle="All candidate models are selected by default — at least one must remain selected"
        action={
          <span className="text-xs font-medium text-slate-400">
            {config.selectedModels.length} of{' '}
            {forecastModels.filter((model) => !unavailableModels[model.id]).length} selected
          </span>
        }
      >
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {forecastModels.map((model) => {
            const isFallback = model.id === config.fallbackModel
            // Disabled rather than hidden: a model missing without
            // explanation reads as a bug, and the reason is worth showing.
            const unavailableReason = unavailableModels[model.id]
            return (
              <Checkbox
                key={model.id}
                checked={!unavailableReason && config.selectedModels.includes(model.id)}
                onChange={(checked) => toggleModel(model.id, checked)}
                disabled={Boolean(unavailableReason)}
                label={
                  unavailableReason
                    ? `${model.name} · unavailable`
                    : isFallback
                      ? `${model.name} · also fallback`
                      : model.name
                }
                description={
                  unavailableReason
                    || (isFallback
                      ? 'Also configured as the fallback baseline for this run.'
                      : model.description)
                }
              />
            )
          })}
        </div>
        {errors?.selectedModels && (
          <p className="mt-3 text-xs text-rose-500">{errors.selectedModels}</p>
        )}
      </SectionContainer>

      <SectionContainer
        title="Default fallback model"
        subtitle="Used only if every evaluated model fails validation"
      >
        <div className="flex items-start gap-3 rounded-xl border border-brand-200 bg-brand-50/60 px-4 py-3.5 dark:border-brand-800 dark:bg-brand-900/20">
          <ShieldQuestion size={18} className="mt-0.5 shrink-0 text-brand-600 dark:text-brand-400" />
          <p className="text-sm text-slate-600 dark:text-slate-300">
            If no model survives ranking, threshold and drift validation, this model produces the
            forecast instead and the run is flagged as having used a fallback.
          </p>
        </div>

        <div className="mt-5 max-w-md">
          <label className="mb-1.5 flex items-center gap-1.5 text-sm font-medium text-slate-600 dark:text-slate-300">
            Fallback Model
            <span className="text-rose-500">*</span>
          </label>
          <Select
            value={config.fallbackModel}
            onChange={selectFallback}
            options={fallbackOptions}
            placeholder="Select fallback model"
            error={!!errors?.fallbackModel}
          />
          <p className="mt-1.5 text-xs text-slate-400">
            Used only if every candidate fails validation. A simple seasonal baseline is
            recommended — it always carries the history's seasonality through the horizon.
          </p>
          {errors?.fallbackModel && (
            <p className="mt-1.5 text-xs text-rose-500">{errors.fallbackModel}</p>
          )}
        </div>
      </SectionContainer>
    </div>
  )
}
