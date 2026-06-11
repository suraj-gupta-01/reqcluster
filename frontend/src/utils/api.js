import axios from 'axios'

const BASE_URL = '/api'

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 300000, // 5 min for long pipeline runs
})

// Backend timestamps are naive UTC (no timezone suffix). new Date() would
// otherwise parse them as local time, so append 'Z' to force UTC parsing.
export const formatTimestamp = (value) => {
  if (!value) return ''
  let iso = String(value)
  const hasTz = /[zZ]|[+-]\d{2}:?\d{2}$/.test(iso)
  if (iso.includes('T') && !hasTz) iso += 'Z'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? String(value) : d.toLocaleString()
}

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

export const getMetrics = async (sessionId) => {
  const res = await api.get(`/metrics?session_id=${sessionId}`)
  return res.data
}

export const getGraph = async (sessionId) => {
  const res = await api.get(`/graph?session_id=${sessionId}`)
  return res.data
}

export const getRequirements = async (sessionId, clusterId = null, params = {}) => {
  let url = `/requirements?session_id=${sessionId}`
  if (clusterId !== null) url += `&cluster_id=${clusterId}`
  
  const config = {}
  if (params && Object.keys(params).length > 0) {
    config.params = {}
    if (params.page !== undefined && params.page !== null) config.params.page = params.page
    if (params.page_size !== undefined && params.page_size !== null) config.params.page_size = params.page_size
    if (params.search !== undefined && params.search !== null) config.params.search = params.search
    if (params.is_noise !== undefined && params.is_noise !== null) config.params.is_noise = params.is_noise
    if (params.sort_field !== undefined && params.sort_field !== null) config.params.sort_field = params.sort_field
    if (params.sort_dir !== undefined && params.sort_dir !== null) config.params.sort_dir = params.sort_dir
  }
  
  const res = await api.get(url, config)
  const data = res.data
  const totalCountHeader = res.headers?.['x-total-count'] || res.headers?.['X-Total-Count']
  if (Array.isArray(data)) {
    data.totalCount = totalCountHeader ? parseInt(totalCountHeader, 10) : data.length
  }
  return data
}

// Phase 3: Refinement helpers

export const generateSuggestions = async (payload) => {
  const res = await api.post('/suggestions/generate', payload)
  return res.data
}

export const getSuggestions = async (sessionId, status = null) => {
  let url = `/suggestions?session_id=${sessionId}`
  if (status) url += `&status=${status}`
  const res = await api.get(url)
  return res.data
}

export const applySuggestion = async (payload) => {
  const res = await api.post('/suggestions/apply', payload)
  return res.data
}

export const getAuditLog = async (sessionId) => {
  const res = await api.get(`/suggestions/audit?session_id=${sessionId}`)
  return res.data
}

// Phase 4: Human-in-the-Loop helpers

export const submitFeedback = async (payload) => {
  const res = await api.post('/feedback/submit', payload)
  return res.data
}

export const getFeedbackQueue = async (sessionId, status = null) => {
  let url = `/feedback/queue?session_id=${sessionId}`
  if (status) url += `&status=${status}`
  const res = await api.get(url)
  return res.data
}

export const reviewFeedback = async (payload) => {
  const res = await api.post('/feedback/review', payload)
  return res.data
}

export const getConstraints = async (sessionId) => {
  const res = await api.get(`/feedback/constraints?session_id=${sessionId}`)
  return res.data
}

export const getFeedbackExportUrl = (sessionId, format = 'csv') => {
  return `${BASE_URL}/feedback/export?session_id=${sessionId}&format=${format}`
}

// DP5: Dependency tree + rationale helpers

export const generateDependencies = async (payload) => {
  const res = await api.post('/dependencies/generate', payload)
  return res.data
}

export const getDependencies = async (sessionId) => {
  const res = await api.get('/dependencies', { params: { session_id: sessionId } })
  return res.data
}

// Phase 5: Active learning helpers

export const runConstrainedClustering = async (sessionId) => {
  const res = await api.post('/cluster/constrained', { session_id: sessionId })
  return res.data
}

export const getUncertaintyQueue = async (sessionId, topK = 20) => {
  const res = await api.get('/active-learning/queue', { params: { session_id: sessionId, top_k: topK } })
  return res.data
}

export const getQualityHistory = async (sessionId) => {
  const res = await api.get('/quality/history', { params: { session_id: sessionId } })
  return res.data
}

// Phase 5: MBSE export. format = reqif | sysml | jama | csv
export const getExportUrl = (sessionId, format = 'reqif') => {
  return `${BASE_URL}/export/${format}?session_id=${sessionId}`
}

export default api

