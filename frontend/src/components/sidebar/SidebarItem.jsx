import { NavLink } from 'react-router-dom'
import * as Icons from 'lucide-react'
import { cn } from '../../utils/cn'

export default function SidebarItem({ label, path, icon, collapsed }) {
  const Icon = Icons[icon] || Icons.Circle

  return (
    <NavLink
      to={path}
      end={path === '/'}
      className={({ isActive }) =>
        cn(
          'group relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-all duration-200',
          isActive
            ? 'bg-brand-600 text-white shadow-md shadow-brand-600/25'
            : 'text-slate-500 hover:bg-slate-100 hover:text-slate-800 dark:text-slate-400 dark:hover:bg-slate-800/70 dark:hover:text-slate-100'
        )
      }
    >
      <Icon size={18} strokeWidth={2} className="shrink-0" />
      {!collapsed && <span className="truncate">{label}</span>}
    </NavLink>
  )
}
