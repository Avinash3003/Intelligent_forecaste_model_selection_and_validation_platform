import { useEffect, useState } from 'react'
import { fetchModelAvailability } from '../services'
import { defaultFallbackModel, defaultForecastHorizon, forecastHorizonRange } from '../data/appConfig'

/**
 * Which candidate models the backend's execution mode can run, and the
 * forecast horizon bounds this platform enforces.
 *
 * Both answer the same question the Configure step needs answered before
 * rendering: "what can this run actually be" — fetched together so the
 * step does not wait on two round trips for one screen.
 *
 * `unavailable` is a `{ [modelId]: reason }` map of models that CANNOT
 * run, so a caller with no entry for a model treats it as available.
 * `horizonRange` is the authoritative `{min, max, default}` in months —
 * the same bounds `POST /deploy` validates against, sourced from
 * `app.config.run_limits` on the backend (see
 * tests/backend/test_run_limits_match_the_engine.py for how that in turn
 * stays in lockstep with the engine's own MIN/MAX_FORECAST_HORIZON).
 * `fallbackModel` is `ModelConfig.DEFAULT_FALLBACK_MODEL` — the model a
 * submitted run actually falls back to when none is chosen, so the picker
 * pre-selects the model that will really run rather than a disconnected
 * guess of what it is.
 *
 * All three default to the local fallback constants in data/appConfig.js —
 * never a fixed value invented here — so a request that fails or has not
 * resolved yet leaves the picker working exactly as it did before this
 * lookup existed, rather than gating the step on a secondary call.
 */
export default function useModelAvailability() {
  const [unavailable, setUnavailable] = useState({})
  const [horizonRange, setHorizonRange] = useState({
    min: forecastHorizonRange.min,
    max: forecastHorizonRange.max,
    default: defaultForecastHorizon,
  })
  const [fallbackModel, setFallbackModel] = useState(defaultFallbackModel)

  useEffect(() => {
    let cancelled = false

    fetchModelAvailability()
      .then((response) => {
        if (cancelled) return
        const blocked = {}
        for (const model of response?.models || []) {
          if (!model.available) blocked[model.id] = model.reason || 'Not available on this execution mode.'
        }
        setUnavailable(blocked)

        const horizon = response?.horizon
        if (horizon && Number.isFinite(horizon.min_months) && Number.isFinite(horizon.max_months)) {
          setHorizonRange({
            min: horizon.min_months,
            max: horizon.max_months,
            default: horizon.default_months ?? defaultForecastHorizon,
          })
        }

        if (response?.default_fallback_model) {
          setFallbackModel(response.default_fallback_model)
        }
      })
      .catch(() => {
        // Availability is an enhancement, never a gate. Staying at the
        // local fallback leaves the picker working exactly as it did
        // before this lookup existed.
      })

    return () => {
      cancelled = true
    }
  }, [])

  return { unavailable, horizonRange, fallbackModel }
}
