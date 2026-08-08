import { useState } from 'react'
import { ChevronsLeft, ChevronsRight, Waypoints } from 'lucide-react'
import SidebarItem from './SidebarItem'
import { sidebarNav, projectInfo } from '../../data/appConfig'
import { cn } from '../../utils/cn'

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <aside
      className={cn(
        'sticky top-0 flex h-screen shrink-0 flex-col border-r border-slate-200/70 bg-white/80 backdrop-blur-xl transition-all duration-300 dark:border-slate-800/70 dark:bg-slate-900/60',
        collapsed ? 'w-[76px]' : 'w-64'
      )}
    >
      <div className="flex items-center gap-2.5 px-4 py-5">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-brand-500 to-indigo-700 text-white shadow-md shadow-brand-600/30">
          <Waypoints size={18} strokeWidth={2.25} />
        </div>
        {!collapsed && (
          <div className="min-w-0">
            <p className="truncate text-sm font-bold text-slate-800 dark:text-slate-100">
              {projectInfo.name}
            </p>
            <p className="truncate text-[11px] text-slate-400">Enterprise AutoML</p>
          </div>
        )}
      </div>

      <nav className="flex-1 space-y-1 px-3 py-2">
        {sidebarNav.map((item) => (
          <SidebarItem key={item.path} {...item} collapsed={collapsed} />
        ))}
      </nav>

      <div className="border-t border-slate-100 p-3 dark:border-slate-800">
        <button
          onClick={() => setCollapsed((c) => !c)}
          className="flex w-full items-center justify-center gap-2 rounded-lg py-2 text-xs font-medium text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600 dark:hover:bg-slate-800 dark:hover:text-slate-300"
        >
          {collapsed ? <ChevronsRight size={16} /> : <ChevronsLeft size={16} />}
          {!collapsed && 'Collapse'}
        </button>
      </div>
    </aside>
  )
}
