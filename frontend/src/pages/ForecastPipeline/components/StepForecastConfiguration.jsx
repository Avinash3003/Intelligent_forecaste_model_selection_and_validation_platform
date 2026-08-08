import { ShieldQuestion } from 'lucide-react'
import SectionContainer from '../../../components/layout/SectionContainer'
import RangeSlider from '../../../components/ui/RangeSlider'
import Checkbox from '../../../components/ui/Checkbox'
import Select from '../../../components/ui/Select'
import { formatMonths, formatMonthsShort } from '../../../utils/formatMonths'
import { forecastModels, forecastHorizonRange, forecastHorizonTicks } from '../../../data/appConfig'

export default function StepForecastConfiguration({ config, onChange, errors }) {
  // The fallback must remain part of the evaluated set, so deselecting it
  // is blocked here rather than silently corrected after the fact.
  const toggleModel = (modelId, checked) => {
    if (!checked && modelId === config.fallbackModel) return
    const next = checked
      ? [...config.selectedModels, modelId]
      : config.selectedModels.filter((id) => id !== modelId)
    onChange('selectedModels', next)
  }

  // Choosing a fallback also selects it for evaluation, so the two can
  // never disagree (the requirement: a fallback is always evaluated too).
  const selectFallback = (modelId) => {
    onChange('fallbackModel', modelId)
    if (!config.selectedModels.includes(modelId)) {
      onChange('selectedModels', [...config.selectedModels, modelId])
    }
  }

  const fallbackOptions = forecastModels
    .filter((model) => config.selectedModels.includes(model.id))
    .map((model) => ({ value: model.id, label: model.name, sublabel: model.description }))

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
            {config.selectedModels.length} of {forecastModels.length} selected
          </span>
        }
      >
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {forecastModels.map((model) => {
            const isFallback = model.id === config.fallbackModel
            return (
              <Checkbox
                key={model.id}
                checked={config.selectedModels.includes(model.id)}
                onChange={(checked) => toggleModel(model.id, checked)}
                disabled={isFallback}
                label={isFallback ? `${model.name} · fallback` : model.name}
                description={
                  isFallback
                    ? 'Selected as the default fallback, so it always participates in evaluation.'
                    : model.description
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
            Only evaluated models can serve as the fallback.
          </p>
          {errors?.fallbackModel && (
            <p className="mt-1.5 text-xs text-rose-500">{errors.fallbackModel}</p>
          )}
        </div>
      </SectionContainer>
    </div>
  )
}
