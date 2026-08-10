import { BrowserRouter } from 'react-router-dom'
import AppRoutes from './routes/AppRoutes'
import { AuthProvider } from './auth/AuthProvider'
import AuthGate from './auth/AuthGate'

export default function App() {
  return (
    <AuthProvider>
      {/* Sign-in wraps the router, not individual routes: an
          unauthenticated browser never renders the app shell at all. */}
      <AuthGate>
        <BrowserRouter>
          <AppRoutes />
        </BrowserRouter>
      </AuthGate>
    </AuthProvider>
  )
}
