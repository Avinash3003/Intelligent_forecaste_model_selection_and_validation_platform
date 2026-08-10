import { API_BASE_URL } from '../services/apiConfig'

/**
 * Entra ID configuration is fetched from the backend at runtime, never
 * baked into this bundle.
 *
 * That is deliberate: a SPA's client id and tenant are public, but they
 * are also *environment-specific*. Building them in would mean one
 * frontend artifact per environment, and a rebuild every time a tenant
 * changed. Fetching them means the same built bundle can be deployed
 * anywhere and takes its identity configuration from whichever backend it
 * is pointed at.
 *
 * No secret is involved on either side — see the backend's
 * `AuthConfigResponse`, which is the only unauthenticated route in the API
 * precisely because the frontend must read it before it can hold a token.
 */
export async function fetchAuthConfig() {
  // A network failure rejects with the browser's own "Failed to fetch",
  // which is meaningless on a sign-in screen. Both failure modes are
  // translated here so the gate never renders a raw platform string.
  let response
  try {
    response = await fetch(`${API_BASE_URL}/auth/config`)
  } catch {
    throw new Error(
      'Could not reach the ForecastIQ service. Check your connection and try again.'
    )
  }
  if (!response.ok) {
    throw new Error('The ForecastIQ service is not responding. Please try again shortly.')
  }
  return response.json()
}

/** MSAL options for a browser SPA using the authorization-code + PKCE flow. */
export function buildMsalConfig({ client_id: clientId, authority }) {
  return {
    auth: {
      clientId,
      authority,
      redirectUri: window.location.origin,
      postLogoutRedirectUri: window.location.origin,
      // The SPA holds no client secret and therefore cannot use a
      // confidential flow; MSAL uses PKCE, which is the correct choice
      // for a public client.
      navigateToLoginRequestUrl: true,
    },
    cache: {
      // Session storage, not local: the sign-in does not outlive the
      // browser tab, so a shared or unattended machine does not keep a
      // usable session lying around.
      cacheLocation: 'sessionStorage',
      storeAuthStateInCookie: false,
    },
  }
}
