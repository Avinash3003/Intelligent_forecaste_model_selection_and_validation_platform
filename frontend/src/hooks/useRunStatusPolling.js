import { useEffect, useRef, useState } from 'react'
import { fetchExecutionStatus, isTerminalStatus } from '../services'

const POLL_INTERVAL_MS = 3000

// Watches one run's execution status until it reaches a terminal state.
//
// Polling rules this enforces, so no caller has to re-implement them:
//   * only polls while the status is non-terminal (Pending / Running)
//   * stops immediately on Completed / Failed / Cancelled
//   * never overlaps requests — the next poll is scheduled only after the
//     previous one settles, so a slow backend cannot queue up calls
//   * cancels cleanly on unmount or when `runId` changes
//
// Returns { status, error, notFound, polling }. `notFound` is surfaced
// separately from `error` because a 404 means "no such run", which the UI
// answers differently from a transport failure.
export function useRunStatusPolling(runId) {
  const [status, setStatus] = useState(null)
  const [error, setError] = useState(null)
  const [notFound, setNotFound] = useState(false)
  const [polling, setPolling] = useState(false)

  // Held in a ref so the cleanup function can cancel an in-flight cycle
  // without the effect needing to re-run on every state change.
  const cancelledRef = useRef(false)
  const timeoutRef = useRef(null)

  useEffect(() => {
    if (!runId) {
      setStatus(null)
      setError(null)
      setNotFound(false)
      setPolling(false)
      return undefined
    }

    cancelledRef.current = false
    setError(null)
    setNotFound(false)
    setPolling(true)

    async function poll() {
      if (cancelledRef.current) return

      try {
        const response = await fetchExecutionStatus(runId)
        if (cancelledRef.current) return

        setStatus(response.job_status)

        if (isTerminalStatus(response.job_status)) {
          setPolling(false)
          return
        }
      } catch (err) {
        if (cancelledRef.current) return

        if (err.status === 404) {
          // The run genuinely does not exist on this backend — polling it
          // again would return 404 forever.
          setNotFound(true)
        } else {
          setError(err.message)
        }
        setPolling(false)
        return
      }

      // Scheduled only after this cycle settled, so requests never overlap.
      timeoutRef.current = setTimeout(poll, POLL_INTERVAL_MS)
    }

    poll()

    return () => {
      cancelledRef.current = true
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
    }
  }, [runId])

  return { status, error, notFound, polling }
}
