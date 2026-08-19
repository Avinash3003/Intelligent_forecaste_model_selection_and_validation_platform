import { useEffect, useMemo, useState } from 'react'
import { formatIST } from '../utils/formatDateTime'

/**
 * The Dataset -> Run selection shared by Results, Experiments and
 * Observability.
 *
 * A dataset accumulates many runs, so a single flat run dropdown becomes
 * unusable fast — the user has to recognise a run by timestamp alone across
 * every dataset at once. Selecting the dataset first narrows the run list to
 * that dataset's own runs, newest first.
 *
 * This lives in one hook rather than being reimplemented per page so the
 * three screens can never disagree about what "newest first" means or about
 * how a run is labelled.
 *
 * @param runs   Normalized deployments (from `fetchDeployments`).
 * @param options.completedOnly  Only offer runs that finished — Experiments
 *   and Observability read artifacts that a running job has not written yet.
 */
export function useDatasetRunFilter(runs, { completedOnly = false, initialRun = '' } = {}) {
  const usable = useMemo(
    () => (completedOnly ? runs.filter((run) => run.status === 'Completed') : runs),
    [runs, completedOnly]
  )

  const [dataset, setDataset] = useState('')
  const [run, setRun] = useState('')

  // Each distinct dataset exactly once, ordered by its most recent run —
  // so the dataset worked on most recently is first, never one entry per run.
  const datasetOptions = useMemo(() => {
    const seen = new Set()
    const options = []
    for (const item of usable) {
      if (item.dataset && !seen.has(item.dataset)) {
        seen.add(item.dataset)
        options.push({ value: item.dataset, label: item.dataset })
      }
    }
    return options
  }, [usable])

  // Only the selected dataset's runs, newest first, labelled by when it ran
  // and its run id — never by the dataset name again, which the dataset
  // dropdown directly above already states.
  const runOptions = useMemo(
    () =>
      usable
        .filter((item) => item.dataset === dataset)
        .map((item) => ({
          value: item.id,
          label: `${formatIST(item.startTimeRaw)} — Run ${item.id}`,
        })),
    [usable, dataset]
  )

  // A run id handed in by the caller (e.g. ?run=... when arriving from the
  // Results page's MLflow card) wins the initial selection, and pulls its
  // own dataset into view so the run is actually reachable in the dropdowns.
  // Applied once: after that the user's own selections take over, so this
  // never fights them.
  const [initialApplied, setInitialApplied] = useState(false)
  useEffect(() => {
    if (initialApplied || !initialRun || !usable.length) return
    const match = usable.find((item) => item.id === initialRun)
    if (match) {
      setDataset(match.dataset)
      setRun(match.id)
    }
    setInitialApplied(true)
  }, [initialRun, usable, initialApplied])

  // Default to the most recent dataset, and re-home the selection if the
  // current one disappears (a filter change, a refreshed run list).
  useEffect(() => {
    // Don't default the dataset out from under a pending initial run.
    if (initialRun && !initialApplied) return
    if (!datasetOptions.length) {
      if (dataset) setDataset('')
      return
    }
    if (!datasetOptions.some((option) => option.value === dataset)) {
      setDataset(datasetOptions[0].value)
    }
  }, [datasetOptions, dataset, initialRun, initialApplied])

  // Keep the run selection valid for whatever dataset is now selected.
  useEffect(() => {
    if (initialRun && !initialApplied) return
    if (!runOptions.length) {
      if (run) setRun('')
      return
    }
    if (!runOptions.some((option) => option.value === run)) {
      setRun(runOptions[0].value)
    }
  }, [runOptions, run, initialRun, initialApplied])

  return { dataset, setDataset, datasetOptions, run, setRun, runOptions }
}

export default useDatasetRunFilter
