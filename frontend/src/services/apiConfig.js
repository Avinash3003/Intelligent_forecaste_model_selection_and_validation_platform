// Single source of truth for the backend base URL. Every API service reads
// from here instead of embedding a URL directly, so switching environments
// (local -> staging -> prod) only ever touches VITE_API_BASE_URL.
//
// The localhost fallback exists purely so `npm run dev` works out of the
// box without requiring a local .env file — it is never relied on outside
// local development. See .env.example.
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export const REQUEST_TIMEOUT_MS = 30000
