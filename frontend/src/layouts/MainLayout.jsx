import { useEffect, useState } from 'react'
import { Outlet } from 'react-router-dom'
import Sidebar from '../components/sidebar/Sidebar'
import Header from '../components/header/Header'

export default function MainLayout() {
  const [darkMode, setDarkMode] = useState(false)

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode)
  }, [darkMode])

  return (
    <div className="flex min-h-screen bg-slate-50 dark:bg-surface">
      <Sidebar />
      <div className="flex min-h-screen flex-1 flex-col">
        <Header darkMode={darkMode} onToggleDarkMode={() => setDarkMode((d) => !d)} />
        <main className="flex-1">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
