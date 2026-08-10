import { useState } from 'react'
import { ChevronsLeft, ChevronsRight } from 'lucide-react'
import SidebarItem from './SidebarItem'
import SigmoidMark from '../brand/SigmoidMark'
import { sidebarNav, projectInfo } from '../../data/appConfig'
import { useAuth } from '../../auth/AuthProvider'
import { cn } from '../../utils/cn'

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false)
  const { can } = useAuth()

  // A link the user would only be refused is worse than no link: it makes
  // a permission problem look like a broken page. Items with no
  // `permission` are open to everyone who can sign in.
  const visibleNav = sidebarNav.filter((item) => !item.permission || can(item.permission))

  return (
    <aside
      className={cn(
        'sticky top-0 flex h-screen shrink-0 flex-col border-r border-slate-200/70 bg-white/80 backdrop-blur-xl transition-all duration-300 dark:border-slate-800/70 dark:bg-slate-900/60',
        // Widened from the original 76px: the larger logo badge (56px)
        // plus its padding no longer fits the old collapsed rail width.
        collapsed ? 'w-24' : 'w-64'
      )}
    >
      <div className="flex items-center gap-3 px-4 py-6">
        {/* Sigmoid's own mark, on a plain badge rather than the brand
            gradient used elsewhere — a logo carries its own color
            identity and should never be retinted to match the page. The
            mark's real aspect ratio is wide (~2.8:1), so the badge hugs
            its width rather than forcing it into a square. */}
        <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-xl border border-slate-200/80 bg-white px-1.5 shadow-sm dark:border-slate-700 dark:bg-slate-100">
          <SigmoidMark size={48} />
        </div>
        {!collapsed && (
          <div className="min-w-0">
            <p className="truncate text-lg font-bold leading-tight text-slate-800 dark:text-slate-100">
              {projectInfo.name}
            </p>
            <p className="truncate text-xs text-slate-400">Enterprise AutoML</p>
          </div>
        )}
      </div>

      <nav className="flex-1 space-y-1 px-3 py-2">
        {visibleNav.map((item) => (
          <SidebarItem key={item.path} {...item} collapsed={collapsed} />
        ))}
      </nav>

      <div className="border-t border-slate-100 p-3 dark:border-slate-800">
        {!collapsed && (
          <p className="mb-2 px-1 text-center text-[10px] font-medium tracking-wide text-slate-300 dark:text-slate-600">
            Powered by Sigmoid
          </p>
        )}
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
