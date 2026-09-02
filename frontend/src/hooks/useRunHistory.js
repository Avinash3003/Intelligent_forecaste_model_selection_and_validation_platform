import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchDeployments, isTerminalStatus } from '../services'

const REFRESH_INTERVAL_MS = 5000

/**
 * The run list, kept current — the single implementation every page that
 * shows run history uses.
 *
 * It exists because of one hazard that is easy to get wrong per page, and
 * was: run history is read from MLflow, and the backend cannot have that
 * ready in the seconds right after a restart. `/deployments` answers `[]`
 * during that window (about eight seconds against the Databricks workspace),
 * which is indistinguishable from "you have never run anything" — and that
 * is exactly what the pages rendered, permanently, because they fetched
 * once and never looked again. Opening Results right after starting the
 * backend reported "No runs yet" to a user with five.
 *
 * So an empty list is always worth another look. It costs nothing once
 * history is warm: the backend answers a repeat list from cache in
 * single-digit milliseconds, and polling stops as soon as there is
 * something to show and nothing still running.
 *
 * Polling continues while any run is non-terminal, so an active run's
 * status stays live without the caller arranging it.
 */
export function useRunHistory() {
  const [runs, setRuns] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const cancelledRef = useRef(false)
  const timeoutRef = useRef(null)

  const load = useCallback(async ({ showSpinner } = {}) => {
    if (showSpinner) setLoading(true)
    try {
      const data = await fetchDeployments()
      if (cancelledRef.current) return data
      setRuns(data)
      setError(null)
      return data
    } catch (err) {
      if (!cancelledRef.current) setError(err.message)
      return null
    } finally {
      if (!cancelledRef.current && showSpinner) setLoading(false)
    }
  }, [])

  useEffect(() => {
    cancelledRef.current = false

    async function cycle(showSpinner) {
      const data = await load({ showSpinner })
      if (cancelledRef.current) return

      const hasActiveRun = (data ?? []).some((run) => !isTerminalStatus(run.status))
      const stillWarming = (data ?? []).length === 0
      if (hasActiveRun || stillWarming) {
        timeoutRef.current = setTimeout(() => cycle(false), REFRESH_INTERVAL_MS)
      }
    }

    cycle(true)

    return () => {
      cancelledRef.current = true
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
    }
  }, [load])

  return { runs, loading, error, reload: load }
}

export { REFRESH_INTERVAL_MS }
