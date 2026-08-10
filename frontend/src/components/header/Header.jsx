import { useEffect, useRef, useState } from 'react'
import { LogOut, Moon, Sun } from 'lucide-react'
import Avatar from '../ui/Avatar'
import { useAuth } from '../../auth/AuthProvider'
import { projectInfo } from '../../data/appConfig'

// Initials from a display name — two words at most, so "Avinash Reddy"
// renders "AR" and a single-word name still renders something.
function initialsOf(name) {
  const parts = (name || '').trim().split(/\s+/).filter(Boolean)
  if (parts.length === 0) return '?'
  return (parts[0][0] + (parts[1]?.[0] ?? '')).toUpperCase()
}

export default function Header({ darkMode, onToggleDarkMode }) {
  const { user, authEnabled, isDevelopmentIdentity, logout } = useAuth()
  const [menuOpen, setMenuOpen] = useState(false)
  const menuRef = useRef(null)

  useEffect(() => {
    if (!menuOpen) return undefined
    const close = (event) => {
      if (!menuRef.current?.contains(event.target)) setMenuOpen(false)
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [menuOpen])

  const displayName = user?.display_name || user?.email || 'Signed in'
  // The role the identity provider asserted. "No role assigned" is shown
  // rather than a default, so an unassigned user sees the actual reason
  // their pages are empty instead of a plausible-looking role.
  const role = user?.roles?.[0] ?? (isDevelopmentIdentity ? 'Local development' : 'No role assigned')

  return (
    <header className="sticky top-0 z-30 flex h-16 items-center gap-4 border-b border-slate-200/70 bg-white/70 px-6 backdrop-blur-xl dark:border-slate-800/70 dark:bg-slate-900/50">
      <div className="flex-1">
        <p className="text-base font-bold text-slate-700 dark:text-slate-200">{projectInfo.name}</p>
        <p className="hidden text-xs text-slate-400 sm:block">{projectInfo.tagline}</p>
      </div>

      {isDevelopmentIdentity && (
        <span className="hidden rounded-md bg-amber-50 px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-amber-600 dark:bg-amber-900/30 dark:text-amber-400 md:inline">
          Auth disabled — local
        </span>
      )}

      <div className="ml-auto flex items-center gap-2">
        <button
          onClick={onToggleDarkMode}
          className="flex h-9 w-9 items-center justify-center rounded-lg text-slate-500 transition-colors hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800"
          aria-label="Toggle theme"
        >
          {darkMode ? <Sun size={18} /> : <Moon size={18} />}
        </button>

        <div className="mx-1 h-6 w-px bg-slate-200 dark:bg-slate-800" />

        <div className="relative" ref={menuRef}>
          <button
            onClick={() => setMenuOpen((open) => !open)}
            className="flex items-center gap-2 rounded-lg px-2 py-1.5 transition-colors hover:bg-slate-100 dark:hover:bg-slate-800"
          >
            <Avatar initials={initialsOf(displayName)} color="bg-brand-600" size="sm" />
            <div className="hidden text-left sm:block">
              <p className="text-xs font-semibold text-slate-700 dark:text-slate-200">{displayName}</p>
              <p className="text-[11px] text-slate-400">{role}</p>
            </div>
          </button>

          {menuOpen && (
            <div className="absolute right-0 top-full z-40 mt-1.5 w-56 overflow-hidden rounded-xl border border-slate-200 bg-white shadow-lg dark:border-slate-800 dark:bg-slate-900">
              <div className="border-b border-slate-100 px-3 py-2.5 dark:border-slate-800">
                <p className="truncate text-xs font-semibold text-slate-700 dark:text-slate-200">{displayName}</p>
                {user?.email && <p className="truncate text-[11px] text-slate-400">{user.email}</p>}
                <p className="mt-1 text-[11px] text-slate-400">{role}</p>
              </div>
              {/* Sign-out only exists when there is a real session to end;
                  with auth disabled locally there is nothing to sign out of. */}
              {authEnabled ? (
                <button
                  onClick={logout}
                  className="flex w-full items-center gap-2 px-3 py-2.5 text-xs font-medium text-slate-600 transition-colors hover:bg-slate-50 dark:text-slate-300 dark:hover:bg-slate-800"
                >
                  <LogOut size={14} /> Sign out
                </button>
              ) : (
                <p className="px-3 py-2.5 text-[11px] text-slate-400">
                  Authentication is disabled in this environment.
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </header>
  )
}
