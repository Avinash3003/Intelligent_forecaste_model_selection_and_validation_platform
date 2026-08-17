import { useEffect, useState } from 'react'
import { fetchModelAvailability } from '../services'

/**
 * Which candidate models the backend's execution mode can run.
 *
 * Returns a `{ [modelId]: reason }` map of models that CANNOT run, so a
 * caller with no entry for a model treats it as available. That default
 * matters: if this request fails the picker keeps working exactly as it
 * did before, offering everything, rather than disabling every model
 * because a secondary lookup was unreachable.
 */
export default function useModelAvailability() {
  const [unavailable, setUnavailable] = useState({})

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
      })
      .catch(() => {
        // Availability is an enhancement, never a gate. Staying empty
        // leaves every model selectable, which is the pre-existing
        // behaviour.
      })

    return () => {
      cancelled = true
    }
  }, [])

  return unavailable
}
