import axios from 'axios'

const BASE_URL = '/api'

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 300000, // 5 min for long pipeline runs
})

export const getErrorMessage = (error, fallback = 'Request failed.') => {
  const detail = error?.response?.data?.detail
  let message = fallback
  if (typeof detail === 'string') message = detail
  else if (Array.isArray(detail)) {
    message = detail
      .map(item => item?.msg || item?.message)
      .filter(Boolean)
      .join(' ')
  } else if (detail?.message) {
    message = detail.message
  } else if (error?.message && !String(error.message).includes('stack')) {
    message = error.message
  }
  return String(message || fallback)
    .replace(/sk-[A-Za-z0-9_-]+/g, '[redacted]')
    .replace(/\s+at\s+.+/g, '')
    .slice(0, 500)
}

export const uploadRequirements = async (file) => {
  const form = new FormData()
  form.append('file', file)
  const res = await api.post('/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return res.data
}

export const runClustering = async (payload) => {
  const res = await api.post('/cluster', payload)
  return res.data
}

export const clusterSession = runClustering

export const enrichSession = async (payload) => {
  const res = await api.post('/enrich', payload)
  return res.data
}

export const getEnrichmentStatus = async (sessionId) => {
  const res = await api.get(`/enrich/status/${sessionId}`)
  return res.data
}

export const getEnrichmentResults = async (sessionId) => {
  const res = await api.get('/enrich/results', { params: { session_id: sessionId } })
  return res.data
}

export const getProgress = async (sessionId) => {
  const res = await api.get(`/progress/${sessionId}`)
  return res.data
}

export const getSessions = async () => {
  const res = await api.get('/sessions')
  return res.data
}

export const getSession = async (sessionId) => {
  const res = await api.get(`/sessions/${sessionId}`)
  return res.data
}

export const getClusters = async (sessionId) => {
  const res = await api.get(`/clusters?session_id=${sessionId}`)
  return res.data
}

export const getClusterDetail = async (sessionId, clusterId) => {
  const res = await api.get(`/cluster/${clusterId}?session_id=${sessionId}`)
  return res.data
}

export const getGraph = async (sessionId) => {
  const res = await api.get(`/graph?session_id=${sessionId}`)
  return res.data
}

export const getRequirements = async (sessionId, clusterId = null) => {
  let url = `/requirements?session_id=${sessionId}`
  if (clusterId !== null) url += `&cluster_id=${clusterId}`
  const res = await api.get(url)
  return res.data
}

export default api
