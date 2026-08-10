import { apiClient } from './apiClient'

/**
 * The signed-in user as the *backend* understands them: identity plus the
 * permissions its own RBAC table grants.
 *
 * Deliberately not derived from the access token client-side. Decoding a
 * JWT in the browser would duplicate the role -> permission mapping in a
 * second place, where it could drift; and it would invite treating that
 * client-side copy as authoritative, which it never is.
 */
export function fetchCurrentUser() {
  return apiClient.get('/auth/me')
}
