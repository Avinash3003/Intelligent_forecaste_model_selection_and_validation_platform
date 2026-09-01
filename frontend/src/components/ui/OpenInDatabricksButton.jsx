import { ExternalLink } from 'lucide-react'

/**
 * "Open with Databricks" — the single implementation of this action.
 *
 * The URL is built server-side from Databricks' own `run_page_url`, so no
 * workspace identifier is compiled into this bundle and no URL shape is
 * guessed. Navigation carries no credential of any kind: Databricks applies
 * the viewer's own SSO session on arrival and shows them exactly what their
 * role permits. A user without workspace access sees Databricks' own
 * sign-in or permission page, never ForecastIQ's service principal.
 *
 * Renders nothing without a URL. The backend sends null whenever it cannot
 * build a correct one — a local run, or a submission Databricks has not
 * accepted yet — so a run that has no link still shows everything else.
 */
export default function OpenInDatabricksButton({
  url,
  label = 'Open with Databricks',
  size = 'md',
  className = '',
}) {
  if (!url) return null

  const sizing =
    size === 'sm' ? 'gap-1.5 px-2.5 py-1.5 text-xs' : 'gap-2 px-3.5 py-2 text-sm'

  return (
    <a
      href={url}
      target="_blank"
      // noopener stops the new tab reaching back through window.opener;
      // noreferrer keeps the ForecastIQ URL out of the Referer header.
      rel="noopener noreferrer"
      title="Open this run in the Databricks workspace. Databricks applies your own sign-in and permissions."
      className={
        'group relative inline-flex shrink-0 items-center rounded-lg font-semibold ' +
        'border border-brand-200/70 bg-gradient-to-b from-white to-slate-50 text-brand-700 ' +
        'shadow-[0_1px_2px_rgba(15,23,42,0.06)] transition-all duration-200 ' +
        'hover:border-brand-400 hover:text-brand-800 hover:shadow-[0_2px_10px_rgba(79,70,229,0.18)] ' +
        'hover:-translate-y-px active:translate-y-0 ' +
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 focus-visible:ring-offset-2 ' +
        'dark:border-slate-700 dark:from-slate-800 dark:to-slate-900 dark:text-brand-300 ' +
        'dark:hover:border-brand-500 dark:hover:text-brand-200 dark:focus-visible:ring-offset-slate-900 ' +
        sizing +
        ' ' +
        className
      }
    >
      <DatabricksMark />
      <span>{label}</span>
      <ExternalLink
        size={size === 'sm' ? 12 : 14}
        className="shrink-0 opacity-60 transition-transform duration-200 group-hover:translate-x-0.5 group-hover:opacity-100"
        aria-hidden="true"
      />
    </a>
  )
}

/** The Databricks stacked-chevron mark, drawn rather than fetched so the
 *  button has no network dependency and inherits the current text colour. */
function DatabricksMark() {
  return (
    <svg
      viewBox="0 0 24 24"
      className="h-3.5 w-3.5 shrink-0"
      fill="currentColor"
      aria-hidden="true"
      focusable="false"
    >
      <path d="M12 2.4 2.6 7.1v2.3L12 14.1l9.4-4.7V7.1L12 2.4Zm0 2 6.6 3.3L12 11 5.4 7.7 12 4.4Z" />
      <path d="M2.6 12.1v2.3L12 19.1l9.4-4.7v-2.3L12 16.8l-9.4-4.7Z" opacity=".75" />
      <path d="M2.6 16.6v2.3L12 23.6l9.4-4.7v-2.3L12 21.3l-9.4-4.7Z" opacity=".5" />
    </svg>
  )
}
