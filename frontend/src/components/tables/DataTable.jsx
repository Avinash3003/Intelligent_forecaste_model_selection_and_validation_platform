import { cn } from '../../utils/cn'

// Generic, reusable table. Columns: [{ key, header, render? }]
// Optional `rowClassName(row)` returns extra classes for a given row (e.g. to highlight it).
export default function DataTable({ columns, data, className, rowClassName }) {
  return (
    <div className={cn('overflow-x-auto rounded-xl border border-slate-100 dark:border-slate-800', className)}>
      <table className="w-full min-w-[640px] text-left text-sm">
        <thead>
          <tr className="border-b border-slate-100 bg-slate-50/60 dark:bg-slate-800/40 dark:border-slate-800">
            {columns.map((col) => (
              <th
                key={col.key}
                className="px-5 py-3 text-xs font-semibold uppercase tracking-wide text-slate-400"
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, idx) => (
            <tr
              key={row.id ?? idx}
              className={cn(
                'border-b border-slate-50 last:border-0 transition-colors hover:bg-slate-50/70 dark:border-slate-800/60 dark:hover:bg-slate-800/40',
                rowClassName?.(row)
              )}
            >
              {columns.map((col) => (
                <td key={col.key} className="px-5 py-3.5 text-slate-600 dark:text-slate-300">
                  {col.render ? col.render(row) : row[col.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
