import { request, downloadUrl } from './client.js'

const s = (sid) => `/sessions/${sid}`
const post = (path, body, o) => request(path, { method: 'POST', body, signal: o?.signal })
const get = (path, query, o) => request(path, { query, signal: o?.signal })

export function uploadSession(file, o) {
  const body = new FormData()
  body.append('file', file)
  return post('/sessions', body, o)
}
export const listSessions = (o) => get('/sessions', undefined, o)
export const getSession = (sid, o) => get(s(sid), undefined, o)
export const deleteSession = (sid, o) => request(s(sid), { method: 'DELETE', signal: o?.signal })

export const startClustering = (sid, params, o) => post(`${s(sid)}/cluster`, params ?? {}, o)
export const getJob = (id, o) => get(`/jobs/${id}`, undefined, o)
export const cancelJob = (id, o) => post(`/jobs/${id}/cancel`, undefined, o)

export const getClusters = (sid, o) => get(`${s(sid)}/clusters`, undefined, o)
export const getClusterDetail = (sid, cid, o) => get(`${s(sid)}/clusters/${cid}`, undefined, o)
export const getGraph = (sid, o) => get(`${s(sid)}/graph`, undefined, o)
export const getMetrics = (sid, o) => get(`${s(sid)}/metrics`, undefined, o)

export async function getRequirements(sid, query, o) {
  const { data, headers } = await request(`${s(sid)}/requirements`, { query, signal: o?.signal, raw: true })
  return { items: data, total: Number(headers.get('X-Total-Count') ?? data.length) }
}

export const startEnrichment = (sid, body, o) => post(`${s(sid)}/enrich`, body ?? {}, o)
export const getEnrichment = (sid, o) => get(`${s(sid)}/enrichment`, undefined, o)

export const startSuggestions = (sid, params, o) => post(`${s(sid)}/suggestions`, params ?? {}, o)
export const getSuggestions = (sid, status, o) => get(`${s(sid)}/suggestions`, { status }, o)
export const decideSuggestion = (sid, id, decision, actor, o) =>
  post(`${s(sid)}/suggestions/${id}/${decision}`, actor ? { actor } : {}, o)
export const getAudit = (sid, o) => get(`${s(sid)}/audit`, undefined, o)

export const submitFeedback = (sid, body, o) => post(`${s(sid)}/feedback`, body, o)
export const getFeedback = (sid, status, o) => get(`${s(sid)}/feedback`, { status }, o)
export const reviewFeedback = (sid, fid, decision, o) => post(`${s(sid)}/feedback/${fid}/${decision}`, undefined, o)
export const getConstraints = (sid, o) => get(`${s(sid)}/constraints`, undefined, o)
export const feedbackExportUrl = (sid, format) => downloadUrl(`${s(sid)}/feedback/export`, { format })

export const startRecluster = (sid, o) => post(`${s(sid)}/recluster`, undefined, o)
export const getUncertainty = (sid, topK, o) => get(`${s(sid)}/uncertainty`, { top_k: topK }, o)
export const getQuality = (sid, o) => get(`${s(sid)}/quality`, undefined, o)

export const startDependencies = (sid, body, o) => post(`${s(sid)}/dependencies`, body ?? {}, o)
export const getDependencies = (sid, o) => get(`${s(sid)}/dependencies`, undefined, o)
export const explainEdge = (sid, body, o) => post(`${s(sid)}/dependencies/explain`, body, o)

export const exportUrl = (sid, fmt) => downloadUrl(`${s(sid)}/export/${fmt}`)
