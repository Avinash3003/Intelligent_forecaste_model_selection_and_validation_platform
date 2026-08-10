import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { PublicClientApplication, InteractionRequiredAuthError } from '@azure/msal-browser'
import { buildMsalConfig, fetchAuthConfig } from './authConfig'
import { setAuthTokenProvider } from '../services/apiClient'
import { fetchCurrentUser } from '../services/authService'

const AuthContext = createContext(null)

// 'loading'         — reading /auth/config, or restoring an existing session
// 'unauthenticated' — Entra sign-in required
// 'authenticated'   — a Principal is available on `user`
// 'error'           — the API or the identity provider could not be reached
const STATUS = { LOADING: 'loading', UNAUTHENTICATED: 'unauthenticated', AUTHENTICATED: 'authenticated', ERROR: 'error' }

/**
 * Entra ID sign-in and the signed-in user's permissions.
 *
 * The access token lives in MSAL's session storage and is attached to API
 * calls by `apiClient`; no component ever touches it, and it is never
 * written into application state or a cookie this app controls.
 *
 * `user.permissions` comes from the *backend*, not from decoding the token
 * client-side. The UI uses it only to hide actions a user would be refused
 * — every one of those actions is independently enforced server-side, so a
 * user who forces a hidden button into view still gets a 403.
 */
export function AuthProvider({ children }) {
  const [status, setStatus] = useState(STATUS.LOADING)
  const [config, setConfig] = useState(null)
  const [user, setUser] = useState(null)
  const [error, setError] = useState(null)
  const msalRef = useRef(null)

  // Acquires a token for the API silently, falling back to a redirect when
  // the session genuinely needs the user again (expired refresh token, MFA,
  // consent). Registered with apiClient so every request carries a fresh
  // token without any caller having to think about expiry.
  const acquireToken = useCallback(async () => {
    const msal = msalRef.current
    if (!msal || !config?.api_scope) return null

    const account = msal.getAllAccounts()[0]
    if (!account) return null

    try {
      const result = await msal.acquireTokenSilent({ scopes: [config.api_scope], account })
      return result.accessToken
    } catch (err) {
      if (err instanceof InteractionRequiredAuthError) {
        await msal.acquireTokenRedirect({ scopes: [config.api_scope], account })
      }
      return null
    }
  }, [config])

  useEffect(() => {
    let cancelled = false

    async function bootstrap() {
      let authConfig
      try {
        authConfig = await fetchAuthConfig()
      } catch (err) {
        if (!cancelled) {
          setError(err.message)
          setStatus(STATUS.ERROR)
        }
        return
      }
      if (cancelled) return
      setConfig(authConfig)

      // Local development: the backend issues a development identity and
      // expects no token at all. It reports `is_development_identity` on
      // that user so the UI can say so out loud.
      if (!authConfig.auth_enabled) {
        setAuthTokenProvider(async () => null)
        try {
          const me = await fetchCurrentUser()
          if (!cancelled) {
            setUser(me)
            setStatus(STATUS.AUTHENTICATED)
          }
        } catch (err) {
          if (!cancelled) {
            setError(err.message)
            setStatus(STATUS.ERROR)
          }
        }
        return
      }

      if (!authConfig.client_id || !authConfig.authority) {
        if (!cancelled) {
          setError('Sign-in is not configured on the server. Contact a ForecastIQ administrator.')
          setStatus(STATUS.ERROR)
        }
        return
      }

      try {
        const msal = new PublicClientApplication(buildMsalConfig(authConfig))
        await msal.initialize()
        // Completes a redirect that is landing right now; returns null on
        // an ordinary page load.
        const redirectResult = await msal.handleRedirectPromise()
        msalRef.current = msal

        const account = redirectResult?.account ?? msal.getAllAccounts()[0]
        if (!account) {
          if (!cancelled) setStatus(STATUS.UNAUTHENTICATED)
          return
        }
        msal.setActiveAccount(account)
      } catch (err) {
        if (!cancelled) {
          setError('Sign-in could not be completed. Please try again.')
          setStatus(STATUS.ERROR)
        }
        return
      }

      setAuthTokenProvider(acquireTokenFor(msalRef.current, authConfig.api_scope))

      try {
        const me = await fetchCurrentUser()
        if (!cancelled) {
          setUser(me)
          setStatus(STATUS.AUTHENTICATED)
        }
      } catch (err) {
        // A valid Entra sign-in the API rejects is the "authenticated but
        // not authorized" case — most often no app role assigned. Saying
        // so is far more useful than bouncing the user back to sign-in.
        if (!cancelled) {
          setError(err.message)
          setStatus(STATUS.ERROR)
        }
      }
    }

    bootstrap()
    return () => {
      cancelled = true
    }
    // Runs once: `acquireToken` is registered separately below, and
    // re-running bootstrap on every config change would restart sign-in.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const login = useCallback(async () => {
    const msal = msalRef.current
    if (!msal || !config?.api_scope) return
    await msal.loginRedirect({
      scopes: [config.api_scope],
      // Development only. In production an existing Microsoft session
      // signing the user straight through is the desired behaviour —
      // forcing an account picker there would add a click to every visit
      // and defeat the point of single sign-on. Locally it is useful for
      // switching between accounts of different roles.
      ...(import.meta.env.DEV ? { prompt: 'select_account' } : {}),
    })
  }, [config])

  const logout = useCallback(async () => {
    const msal = msalRef.current
    if (!msal) return
    await msal.logoutRedirect()
  }, [])

  const value = useMemo(() => {
    const permissions = user?.permissions ?? []
    return {
      status,
      error,
      user,
      authEnabled: Boolean(config?.auth_enabled),
      isDevelopmentIdentity: Boolean(user?.is_development_identity),
      permissions,
      can: (permission) => permissions.includes(permission),
      login,
      logout,
      acquireToken,
    }
  }, [status, error, user, config, login, logout, acquireToken])

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

// Bound token provider handed to apiClient — kept outside the component so
// it closes over the MSAL instance rather than over React state, and so a
// re-render can never leave apiClient holding a stale provider.
function acquireTokenFor(msal, scope) {
  return async () => {
    const account = msal.getActiveAccount() ?? msal.getAllAccounts()[0]
    if (!account || !scope) return null
    try {
      const result = await msal.acquireTokenSilent({ scopes: [scope], account })
      return result.accessToken
    } catch (err) {
      if (err instanceof InteractionRequiredAuthError) {
        await msal.acquireTokenRedirect({ scopes: [scope], account })
      }
      return null
    }
  }
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) throw new Error('useAuth must be used inside an AuthProvider.')
  return context
}

export { STATUS as AUTH_STATUS }
