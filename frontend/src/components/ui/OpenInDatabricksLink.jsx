import { ExternalLink } from 'lucide-react'

/**
 * "Open in Databricks ↗" — opens this run in the Databricks MLflow UI.
 *
 * The URL is built by the backend (app/services/databricks_links.py) from the
 * configured workspace host plus the run's experiment/run ids, so no
 * workspace identifier is compiled into this bundle. When the backend cannot
 * build a correct URL — no host configured, missing ids, or a run tracked to
 * a local store — it sends null and this renders nothing, leaving the rest of
 * the MLflow record intact.
 *
 * Phase 1 is navigation only: Databricks applies its own session/SSO on
 * arrival. No credential is involved here.
 */
export default function OpenInDatabricksLink({ url, className = '' }) {
  if (!url) return null

  return (
    <a
      href={url}
      target="_blank"
      // noopener keeps the new tab from reaching back into this one via
      // window.opener; noreferrer stops the ForecastIQ URL leaking as a
      // Referer. Both are required for a cross-origin _blank link.
      rel="noopener noreferrer"
      title="Open this run in the Databricks workspace (requires Databricks access; use 'View run details' if you do not have it)"
      className={
        'inline-flex items-center gap-1.5 rounded-lg border border-slate-200 px-2.5 py-1.5 ' +
        'text-xs font-semibold text-slate-600 transition hover:border-brand-300 hover:text-brand-600 ' +
        'dark:border-slate-700 dark:text-slate-300 dark:hover:border-brand-500 dark:hover:text-brand-400 ' +
        className
      }
    >
      Open in Databricks
      <ExternalLink size={13} className="shrink-0" aria-hidden="true" />
    </a>
  )
}
