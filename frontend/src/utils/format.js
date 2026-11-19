// Timestamps without a zone are UTC; new Date() would read them as local time, so add 'Z'.
export const formatTimestamp = (value) => {
  if (!value) return ''
  let iso = String(value)
  const hasTz = /[zZ]|[+-]\d{2}:?\d{2}$/.test(iso)
  if (iso.includes('T') && !hasTz) iso += 'Z'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? String(value) : d.toLocaleString()
}

// ApiError (api/client.js) already carries the server's envelope message.
export const getErrorMessage = (error, fallback = 'Request failed.') => {
  const message = (error?.name !== 'AbortError' && error?.message) || fallback
  return String(message)
    .replace(/sk-[A-Za-z0-9_-]+/g, '[redacted]')
    .replace(/\s+at\s+.+/g, '')
    .slice(0, 500)
}

// A 409 means the data changed underneath the user (stale suggestion, concurrent edit, running job).
export const isConflict = (error) => error?.status === 409

// For optional data that is legitimately absent until a job ran (409): use the fallback;
// any other error still reaches useApi's `error` so the page can show it.
export const conflictAs = (fallback) => (error) => {
  if (isConflict(error)) return fallback
  throw error
}
