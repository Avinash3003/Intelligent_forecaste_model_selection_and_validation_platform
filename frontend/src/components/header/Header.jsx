import { Moon, Sun } from 'lucide-react'
import Avatar from '../ui/Avatar'
import { currentUser, projectInfo } from '../../data/appConfig'

export default function Header({ darkMode, onToggleDarkMode }) {
  return (
    <header className="sticky top-0 z-30 flex h-16 items-center gap-4 border-b border-slate-200/70 bg-white/70 px-6 backdrop-blur-xl dark:border-slate-800/70 dark:bg-slate-900/50">
      <div className="flex-1">
        <p className="text-sm font-semibold text-slate-700 dark:text-slate-200">{projectInfo.name}</p>
        <p className="hidden text-[11px] text-slate-400 sm:block">{projectInfo.tagline}</p>
      </div>

      <div className="ml-auto flex items-center gap-2">
        <button
          onClick={onToggleDarkMode}
          className="flex h-9 w-9 items-center justify-center rounded-lg text-slate-500 transition-colors hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800"
          aria-label="Toggle theme"
        >
          {darkMode ? <Sun size={18} /> : <Moon size={18} />}
        </button>

        <div className="mx-1 h-6 w-px bg-slate-200 dark:bg-slate-800" />

        <button className="flex items-center gap-2 rounded-lg px-2 py-1.5 transition-colors hover:bg-slate-100 dark:hover:bg-slate-800">
          <Avatar initials={currentUser.initials} color={currentUser.avatarColor} size="sm" />
          <div className="hidden text-left sm:block">
            <p className="text-xs font-semibold text-slate-700 dark:text-slate-200">{currentUser.name}</p>
            <p className="text-[11px] text-slate-400">{currentUser.role}</p>
          </div>
        </button>
      </div>
    </header>
  )
}
