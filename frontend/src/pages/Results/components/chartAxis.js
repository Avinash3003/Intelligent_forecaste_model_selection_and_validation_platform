/**
 * Axis-label logic for the Actual vs Forecast chart.
 *
 * Pure functions, kept out of the component so the chart stays
 * presentational and this stays testable on its own.
 *
 * The one rule everything here follows: THIN THE LABELS, NEVER THE DATA.
 * Every point the backend sends is plotted and connected; these functions
 * only decide which of them get a visible tick, and how that tick reads.
 */

const MS_PER_DAY = 1000 * 60 * 60 * 24

/** ISO dates ("2015-01-01") are labelled as dates; "T1" and friends are not. */
export function isDateLike(value) {
  return typeof value === 'string' && /^\d{4}-\d{2}-\d{2}/.test(value)
}

export function parseDatePeriod(value) {
  if (!isDateLike(value)) return null
  const date = new Date(value.slice(0, 10))
  return Number.isNaN(date.getTime()) ? null : date
}

/**
 * A point's axis text: its projected calendar date when the backend could
 * derive one, otherwise its own identity ("T1") so nothing renders blank.
 */
export function pointLabel(point) {
  return point?.label || point?.period || ''
}

/**
 * Label granularity, chosen from the span the data actually covers rather
 * than from an assumption about its grain. A two-month upload reads as
 * "Jan 5"; five years of months read as "Jan 15"; multi-decade as "2015".
 */
export function chooseGranularity(spanDays) {
  if (spanDays > 365 * 8) return 'year'
  if (spanDays > 70) return 'month'
  return 'day'
}

const FORMATTERS = {
  year: { year: 'numeric' },
  month: { month: 'short', year: '2-digit' },
  day: { month: 'short', day: 'numeric' },
}

export function formatLabel(value, granularity) {
  const date = parseDatePeriod(value)
  if (!date) return value || ''
  return date.toLocaleDateString('en-US', FORMATTERS[granularity] || FORMATTERS.month)
}

export function formatTooltipLabel(value) {
  const date = parseDatePeriod(value)
  return date
    ? date.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })
    : value || ''
}

/** Total days the actual history covers — 0 when there is nothing to span. */
export function historySpanDays(points) {
  const times = points
    .filter((point) => point.actual != null)
    .map((point) => parseDatePeriod(pointLabel(point)))
    .filter(Boolean)
    .map((date) => date.getTime())
  return times.length > 1 ? (Math.max(...times) - Math.min(...times)) / MS_PER_DAY : 0
}

/**
 * Which points get a visible axis tick.
 *
 * Two passes, in this order, because the second is meaningless without the
 * first:
 *
 *  1. DEDUPLICATE. At year granularity, sixty monthly points produce sixty
 *     ticks reading "2013, 2013, 2013…". A tick is kept only when its text
 *     differs from the last kept one, so each year appears once. This is
 *     what an evenly-spaced `interval` cannot do — it picks positions, not
 *     distinct labels, which is exactly how the axis ended up reading
 *     "2013 2013 2014 2015 2015 2016 2017 2017".
 *
 *  2. SUBSAMPLE. If distinct labels still outnumber what fits, keep an
 *     evenly spread subset — but always the first, the last, and the
 *     history/forecast boundary, since those three carry the most meaning.
 *
 * Returns the `period` values recharts should draw ticks for.
 */
export function selectTicks(points, granularity, maxTicks) {
  if (!points.length) return []

  const boundaryIndex = points.findIndex((point) => point.boundary)

  const distinct = []
  let lastText = null
  points.forEach((point, index) => {
    const text = formatLabel(pointLabel(point), granularity)
    if (text && text !== lastText) {
      distinct.push(index)
      lastText = text
    }
  })

  // A boundary whose label collided with its neighbour is still worth a
  // tick — it is the one position the reader needs to locate.
  if (boundaryIndex >= 0 && !distinct.includes(boundaryIndex)) {
    distinct.push(boundaryIndex)
    distinct.sort((a, b) => a - b)
  }

  if (distinct.length <= maxTicks) {
    return distinct.map((index) => points[index].period)
  }

  const mustKeep = new Set([distinct[0], distinct[distinct.length - 1]])
  if (boundaryIndex >= 0) mustKeep.add(boundaryIndex)

  const budget = Math.max(maxTicks - mustKeep.size, 1)
  const step = distinct.length / budget
  const kept = new Set(mustKeep)
  for (let slot = 0; slot < budget; slot += 1) {
    kept.add(distinct[Math.min(Math.floor(slot * step), distinct.length - 1)])
  }

  return [...kept].sort((a, b) => a - b).map((index) => points[index].period)
}
