import { API_BASE_URL, REQUEST_TIMEOUT_MS, UPLOAD_STALL_TIMEOUT_MS } from './apiConfig'

// Thrown for every failure mode (HTTP error, network failure, timeout) so
// calling code only ever has to handle one error type and can always show
// `error.message` directly to the user — never a raw exception.
export class ApiError extends Error {
  constructor(message, { status, isTimeout = false } = {}) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    // A client-side abort because the response took too long — distinct
    // from every other failure mode. A large-but-legitimate payload that
    // is still loading is not the same situation as a 404/500/network
    // failure, and callers that show a message need to tell them apart
    // (see Results.jsx, whose Results page can genuinely take longer than
    // the default timeout on a large run's first, cold-cache load).
    this.isTimeout = isTimeout
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

// Supplies the Entra ID access token for outgoing requests. Registered
// once by AuthProvider; until then (and in local development, where the
// backend issues a development identity) it returns null and requests go
// out unauthenticated.
//
// A function rather than a stored token on purpose: MSAL refreshes tokens
// behind the scenes, so asking for one per request always gets a live
// token, where caching a string here would eventually send an expired one.
let authTokenProvider = async () => null

export function setAuthTokenProvider(provider) {
  authTokenProvider = provider
}

async function authHeaders() {
  try {
    const token = await authTokenProvider()
    return token ? { Authorization: `Bearer ${token}` } : {}
  } catch {
    // A token that cannot be acquired must not take the request down here
    // — it goes out unauthenticated and the API answers 401, which the
    // caller already knows how to show.
    return {}
  }
}

async function request(path, { method = 'GET', body, isFormData = false, timeoutMs = REQUEST_TIMEOUT_MS } = {}) {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs)

  // FormData must set its own multipart boundary, so Content-Type is
  // omitted for uploads — but the Authorization header applies to both.
  const headers = {
    ...(isFormData ? {} : { 'Content-Type': 'application/json' }),
    ...(await authHeaders()),
  }

  let response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method,
      headers,
      body: isFormData ? body : body !== undefined ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    })
  } catch (error) {
    clearTimeout(timeoutId)
    if (error.name === 'AbortError') {
      throw new ApiError('The request took too long to respond. Please try again.', { isTimeout: true })
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

// Uploads are sent with XMLHttpRequest, not fetch, for two reasons.
//
// A total-duration timeout is the wrong model for an upload: how long it
// takes is the file size divided by the user's upstream bandwidth, and
// neither is knowable here. A fixed 30s aborted a 16 MB dataset whenever the
// connection was slower than about 5 Mbit/s, which is why the same file
// uploaded on one attempt and failed on the next. The timer below measures
// *inactivity* instead — it resets every time bytes actually move, so a
// large file on a slow link succeeds while a genuinely stalled connection
// still fails promptly.
//
// fetch also cannot report upload progress at all; XHR can, which is what
// lets the caller show how far along a large dataset is.
async function upload(path, formData, { onProgress, stallTimeoutMs = UPLOAD_STALL_TIMEOUT_MS } = {}) {
  const headers = await authHeaders()

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    let stallTimer

    const armStallTimer = () => {
      clearTimeout(stallTimer)
      stallTimer = setTimeout(() => {
        xhr.abort()
        reject(
          new ApiError(
            'The upload stopped making progress. Please check your connection and try again.',
            { isTimeout: true }
          )
        )
      }, stallTimeoutMs)
    }

    const settle = (finish) => {
      clearTimeout(stallTimer)
      finish()
    }

    xhr.open('POST', `${API_BASE_URL}${path}`)
    for (const [name, value] of Object.entries(headers)) xhr.setRequestHeader(name, value)

    xhr.upload.onprogress = (event) => {
      armStallTimer()
      if (onProgress && event.lengthComputable) onProgress(event.loaded / event.total)
    }
    // Body sent; the server is now reading it, which reports no progress.
    xhr.upload.onload = () => armStallTimer()

    xhr.onload = () =>
      settle(() => {
        let payload = null
        try {
          payload = JSON.parse(xhr.responseText)
        } catch {
          payload = null
        }
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(payload)
          return
        }
        reject(new ApiError(extractErrorMessage(payload), { status: xhr.status }))
      })

    xhr.onerror = () =>
      settle(() =>
        reject(new ApiError('Unable to reach the server. Please check your connection and try again.'))
      )
    xhr.onabort = () => clearTimeout(stallTimer)

    armStallTimer()
    xhr.send(formData)
  })
}

export const apiClient = {
  get: (path, { timeoutMs } = {}) => request(path, { method: 'GET', timeoutMs }),
  post: (path, body, { timeoutMs } = {}) => request(path, { method: 'POST', body, timeoutMs }),
  postForm: (path, formData, options) => upload(path, formData, options),
}
