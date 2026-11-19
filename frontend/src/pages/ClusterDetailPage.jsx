import { useState } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { ChevronLeft, Loader, Tag } from 'lucide-react'
import { getClusterDetail, getClusters, submitFeedback } from '../api/endpoints.js'
import { useApi } from '../hooks/useApi.js'
import { useSession } from '../state/SessionContext.jsx'
import { getErrorMessage } from '../utils/format.js'
import SessionGate from '../components/SessionGate.jsx'
import { getClusterColor } from '../utils/colors.js'
import MoveToClusterModal from '../components/MoveToClusterModal.jsx'

function MembershipBar({ prob }) {
  const pct = Math.round((prob ?? 0) * 100)
  const color = pct >= 80 ? '#10b981' : pct >= 50 ? '#f59e0b' : '#ef4444'
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="text-xs font-mono text-gray-400 w-8 text-right">{pct}%</span>
    </div>
  )
}

export default function ClusterDetailPage() {
  return <SessionGate><ClusterDetail /></SessionGate>
}

function ClusterDetail() {
  const { clusterId } = useParams()
  const { sessionId, reloadSession } = useSession()
  const navigate = useNavigate()

  const detailQ = useApi((o) => getClusterDetail(sessionId, clusterId, o), [sessionId, clusterId])
  const clustersQ = useApi((o) => getClusters(sessionId, o), [sessionId])
  const detail = detailQ.data
  const allClusters = clustersQ.data || []
  const loading = (detailQ.loading && !detail) || (clustersQ.loading && !clustersQ.data)
  const error = detailQ.error || clustersQ.error
  const [search, setSearch] = useState('')
  const [sortBy, setSortBy] = useState('prob_desc')

  const [selectedRequirement, setSelectedRequirement] = useState(null)
  const [isModalOpen, setIsModalOpen] = useState(false)

  const handleReassignSubmit = async (payload) => {
    if (!selectedRequirement) return
    try {
      await submitFeedback(sessionId, { requirement_id: selectedRequirement.id, ...payload })
      detailQ.reload()
      clustersQ.reload()
      reloadSession()
      setIsModalOpen(false)
      setSelectedRequirement(null)
    } catch (err) {
      alert(getErrorMessage(err, 'Failed to reassign cluster.'))
    }
  }

  const filtered = (detail?.requirements || [])
    .filter(r => !search || r.text.toLowerCase().includes(search.toLowerCase()) || r.req_id?.includes(search))
    .sort((a, b) => {
      if (sortBy === 'prob_desc') return (b.probability ?? 0) - (a.probability ?? 0)
      if (sortBy === 'prob_asc') return (a.probability ?? 0) - (b.probability ?? 0)
      if (sortBy === 'id') return (a.req_id || '').localeCompare(b.req_id || '')
      return 0
    })

  if (loading) return (
    <div className="flex items-center justify-center h-64">
      <Loader size={24} className="animate-spin text-brand-400" />
    </div>
  )
  if (error) return <div className="p-8 text-red-400">{getErrorMessage(error, 'Failed to load cluster.')}</div>

  const cluster = detail
  const color = getClusterColor(cluster?.cluster_id)

  // Navigate to prev/next cluster
  const currentIdx = allClusters.findIndex(c => c.cluster_id === cluster?.cluster_id)
  const prevCluster = currentIdx > 0 ? allClusters[currentIdx - 1] : null
  const nextCluster = currentIdx < allClusters.length - 1 ? allClusters[currentIdx + 1] : null

  return (
    <div className="p-8">
      {/* Back nav */}
      <Link
        to={`/s/${sessionId}/overview`}
        className="inline-flex items-center gap-1.5 text-sm text-gray-500 hover:text-gray-300 mb-6 transition-colors"
      >
        <ChevronLeft size={14} />
        Back to Overview
      </Link>

      {/* Cluster header */}
      <div className="card p-6 mb-6">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-4">
            <div
              className="w-12 h-12 rounded-xl flex items-center justify-center text-lg font-bold text-white"
              style={{ backgroundColor: color + '33', border: `2px solid ${color}` }}
            >
              {cluster?.cluster_id}
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">{cluster?.label}</h1>
              <div className="flex items-center gap-3 mt-1">
                <span className="text-sm text-gray-400">{cluster?.size} requirements</span>
                <span className="text-gray-600">·</span>
                <span className="text-sm text-gray-400">Cluster {cluster?.cluster_id}</span>
              </div>
            </div>
          </div>

          {/* Prev / Next */}
          <div className="flex gap-2">
            {prevCluster && (
              <button
                onClick={() => navigate(`/s/${sessionId}/clusters/${prevCluster.cluster_id}`)}
                className="btn-secondary text-xs py-1 px-3"
              >
                ← Previous
              </button>
            )}
            {nextCluster && (
              <button
                onClick={() => navigate(`/s/${sessionId}/clusters/${nextCluster.cluster_id}`)}
                className="btn-secondary text-xs py-1 px-3"
              >
                Next →
              </button>
            )}
          </div>
        </div>

        {/* Keywords */}
        {cluster?.keywords?.length > 0 && (
          <div className="mt-5">
            <p className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2.5">Top Keywords (c-TF-IDF)</p>
            <div className="flex flex-wrap gap-2">
              {cluster.keywords.map((kw, i) => (
                <div
                  key={kw}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium"
                  style={{
                    backgroundColor: color + '20',
                    border: `1px solid ${color}40`,
                    color: i === 0 ? color : '#d1d5db',
                  }}
                >
                  <Tag size={11} />
                  {kw}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Requirements */}
      <div className="card">
        <div className="px-5 py-3 border-b border-gray-800 flex items-center gap-4">
          <h2 className="text-sm font-semibold text-gray-300 flex-1">Requirements</h2>
          <input
            type="text"
            placeholder="Search…"
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="input text-sm py-1.5 w-48"
          />
          <select
            value={sortBy}
            onChange={e => setSortBy(e.target.value)}
            className="input text-sm py-1.5"
          >
            <option value="prob_desc">Membership ↓</option>
            <option value="prob_asc">Membership ↑</option>
            <option value="id">ID</option>
          </select>
        </div>

        <div className="divide-y divide-gray-800/50">
          {filtered.length === 0 ? (
            <div className="py-8 text-center text-gray-500 text-sm">No requirements match your search.</div>
          ) : (
            filtered.map(r => (
              <div key={r.id} className="px-5 py-4 hover:bg-gray-800/40 transition-colors">
                <div className="flex items-start gap-4">
                  <span className="font-mono text-xs text-gray-500 mt-0.5 flex-shrink-0 w-20">{r.req_id}</span>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-gray-200 leading-relaxed">{r.text}</p>
                  </div>
                  <div className="flex items-center gap-4 flex-shrink-0">
                    <div className="w-28 pt-1">
                      <MembershipBar prob={r.probability} />
                    </div>
                    <button
                      type="button"
                      onClick={() => {
                        setSelectedRequirement({ ...r, cluster_id: cluster?.cluster_id })
                        setIsModalOpen(true)
                      }}
                      className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 text-xs transition-colors"
                    >
                      Reassign
                    </button>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      <MoveToClusterModal
        isOpen={isModalOpen}
        onClose={() => {
          setIsModalOpen(false)
          setSelectedRequirement(null)
        }}
        requirement={selectedRequirement}
        clusters={allClusters}
        onSubmit={handleReassignSubmit}
      />
    </div>
  )
}
