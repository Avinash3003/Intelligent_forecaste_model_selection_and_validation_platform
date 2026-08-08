import { API_BASE_URL, REQUEST_TIMEOUT_MS } from './apiConfig'

// Thrown for every failure mode (HTTP error, network failure, timeout) so
// calling code only ever has to handle one error type and can always show
// `error.message` directly to the user — never a raw exception.
export class ApiError extends Error {
  constructor(message, { status } = {}) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

function extractErrorMessage(payload) {
  if (!payload) return 'Something went wrong. Please try again.'
  if (typeof payload.detail === 'string') return payload.detail
  if (Array.isArray(payload.detail)) return 'The request was invalid. Please check your input and try again.'
  if (typeof payload.message === 'string') return payload.message
  return 'Something went wrong. Please try again.'
}

async function parseJsonSafely(response) {
  try {
    return await response.json()
  } catch {
    return null
  }
}

async function request(path, { method = 'GET', body, isFormData = false } = {}) {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)

  let response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method,
      headers: isFormData ? undefined : { 'Content-Type': 'application/json' },
      body: isFormData ? body : body !== undefined ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    })
  } catch (error) {
    clearTimeout(timeoutId)
    if (error.name === 'AbortError') {
      throw new ApiError('The request took too long to respond. Please try again.')
    }
    throw new ApiError('Unable to reach the server. Please check your connection and try again.')
  }
  clearTimeout(timeoutId)

  const payload = await parseJsonSafely(response)

  if (!response.ok) {
    throw new ApiError(extractErrorMessage(payload), { status: response.status })
  }

  return payload
}

export const apiClient = {
  get: (path) => request(path, { method: 'GET' }),
  post: (path, body) => request(path, { method: 'POST', body }),
  postForm: (path, formData) => request(path, { method: 'POST', body: formData, isFormData: true }),
}
