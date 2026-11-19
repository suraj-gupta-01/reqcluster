const BASE = `${import.meta.env.VITE_API_URL ?? ''}/api`
const API_KEY = import.meta.env.VITE_API_KEY

export class ApiError extends Error {
  constructor(status, code, message) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
  }
}

function buildUrl(path, query) {
  const params = new URLSearchParams()
  for (const [k, v] of Object.entries(query || {})) {
    if (v !== undefined && v !== null && v !== '') params.append(k, String(v))
  }
  const qs = params.toString()
  return `${BASE}${path}${qs ? `?${qs}` : ''}`
}

export async function request(path, { method = 'GET', body, query, signal, raw = false } = {}) {
  const headers = {}
  if (API_KEY) headers['X-API-Key'] = API_KEY
  let payload
  if (body instanceof FormData) payload = body
  else if (body !== undefined) {
    headers['Content-Type'] = 'application/json'
    payload = JSON.stringify(body)
  }
  const res = await fetch(buildUrl(path, query), { method, headers, body: payload, signal })
  if (!res.ok) {
    let env = null
    try { env = await res.json() } catch { /* non-JSON error body */ }
    throw new ApiError(res.status, env?.error ?? 'http_error', env?.message ?? res.statusText)
  }
  const data = res.status === 204 ? null : await res.json()
  return raw ? { data, headers: res.headers } : data
}

// Plain links cannot send X-API-Key, so downloads only work when the API needs no key.
export const downloadUrl = (path, query) => buildUrl(path, query)

export async function download(path, filename, query) {
  const headers = {}
  if (API_KEY) headers['X-API-Key'] = API_KEY
  const res = await fetch(buildUrl(path, query), { headers })
  if (!res.ok) {
    let env = null
    try { env = await res.json() } catch { /* non-JSON error body */ }
    throw new ApiError(res.status, env?.error ?? 'http_error', env?.message ?? res.statusText)
  }
  const url = URL.createObjectURL(await res.blob())
  const a = document.createElement('a')
  a.href = url
  // the server names the file (right extension per format); `filename` is only a fallback
  const served = /filename="?([^";]+)"?/.exec(res.headers.get('Content-Disposition') || '')
  a.download = served?.[1] || filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}
