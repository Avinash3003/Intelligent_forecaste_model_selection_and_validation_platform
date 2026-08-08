// Renders a month count as a human-readable duration, e.g.
//   12 -> "1 year", 13 -> "1 year 1 month", 60 -> "5 years".
// Used wherever a forecast horizon is shown, so the same value never
// reads differently in two places.
export function formatMonths(totalMonths) {
  const months = Number(totalMonths) || 0
  const years = Math.floor(months / 12)
  const remainder = months % 12

  const parts = []
  if (years > 0) parts.push(`${years} ${years === 1 ? 'year' : 'years'}`)
  if (remainder > 0) parts.push(`${remainder} ${remainder === 1 ? 'month' : 'months'}`)

  return parts.length > 0 ? parts.join(' ') : '0 months'
}

// Compact variant for tick labels and tight spaces: 12 -> "1y", 18 -> "1y 6m".
export function formatMonthsShort(totalMonths) {
  const months = Number(totalMonths) || 0
  const years = Math.floor(months / 12)
  const remainder = months % 12

  const parts = []
  if (years > 0) parts.push(`${years}y`)
  if (remainder > 0) parts.push(`${remainder}m`)

  return parts.length > 0 ? parts.join(' ') : '0m'
}
