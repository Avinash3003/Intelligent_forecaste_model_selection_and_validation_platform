// Single source of truth for displaying a timestamp anywhere in the app.
//
// The backend always sends raw ISO 8601 UTC (e.g. "2026-08-07T15:24:00Z")
// — no timezone conversion or locale formatting happens server-side, so
// every screen is guaranteed to agree with every other screen. This is the
// one place that conversion happens: IST (Asia/Kolkata, UTC+5:30), 24-hour
// clock, "07 Aug 2026, 15:24 IST".
const IST_TIME_ZONE = 'Asia/Kolkata'

const dateFormatter = new Intl.DateTimeFormat('en-GB', {
  timeZone: IST_TIME_ZONE,
  day: '2-digit',
  month: 'short',
  year: 'numeric',
})

const timeFormatter = new Intl.DateTimeFormat('en-GB', {
  timeZone: IST_TIME_ZONE,
  hour: '2-digit',
  minute: '2-digit',
  hour12: false,
})

// "07 Aug 2026, 15:24 IST" — the standard full display form.
export function formatIST(isoString) {
  if (!isoString) return '—'
  const date = new Date(isoString)
  if (Number.isNaN(date.getTime())) return '—'
  return `${dateFormatter.format(date)}, ${timeFormatter.format(date)} IST`
}

// "15:24 IST" — time only, for dense contexts like a stage timeline where
// the date is already implied by the run.
export function formatISTTime(isoString) {
  if (!isoString) return '—'
  const date = new Date(isoString)
  if (Number.isNaN(date.getTime())) return '—'
  return `${timeFormatter.format(date)} IST`
}

// Whole seconds between two ISO timestamps, or null if either is missing —
// used for stage durations, computed client-side from real started_at/
// completed_at pairs rather than trusting a pre-formatted string.
export function secondsBetween(startIso, endIso) {
  if (!startIso || !endIso) return null
  const start = new Date(startIso).getTime()
  const end = new Date(endIso).getTime()
  if (Number.isNaN(start) || Number.isNaN(end)) return null
  return Math.max(0, (end - start) / 1000)
}

// "4 min" / "12 sec" — a short duration, honest about missing data as
// "—" rather than "0 sec".
export function formatDuration(seconds) {
  if (seconds == null) return '—'
  if (seconds < 60) return `${Math.round(seconds)} sec`
  return `${Math.round(seconds / 60)} min`
}

// "Forecast Run · 07 Aug 2026, 15:24 IST" — a readable label for a run,
// derived from when it started. The real run_id is never replaced by
// this, only supplemented: every screen that shows this label keeps the
// run_id visible too (a smaller subtext, a separate "Run ID" field), since
// the id is still what a user searches by and what correlates with MLflow.
//
// A date-based label was chosen over sequential numbering ("Run #12"):
// numbering would need to be assigned by position in history, which shifts
// meaning if the visible history ever changes (e.g. after a store reset,
// as happened once already in this project) — a timestamp is unambiguous
// and never needs recomputing.
export function formatRunLabel(startedAtIso) {
  return `Forecast Run · ${formatIST(startedAtIso)}`
}

const monthYearFormatter = new Intl.DateTimeFormat('en-US', { month: 'short', year: 'numeric' })

// "Jan 2021 → Feb 2025" — a dataset's own observed date coverage, always
// month+year even when the underlying values are full timestamps (a
// dataset's coverage is never meaningfully about time-of-day). Neither
// bound is ever guessed: "Date range unavailable" when either is missing
// or unparsable, matching the honest-about-missing-data convention the
// rest of this file follows ("—" for a single value).
export function formatDateRange(startIso, endIso) {
  if (!startIso || !endIso) return 'Date range unavailable'
  const start = new Date(startIso)
  const end = new Date(endIso)
  if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) return 'Date range unavailable'
  return `${monthYearFormatter.format(start)} → ${monthYearFormatter.format(end)}`
}
