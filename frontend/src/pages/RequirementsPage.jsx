import { useState, useEffect, useMemo } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Loader, Search, ChevronRight, ArrowUpDown, Sparkles, AlertTriangle, ChevronDown } from 'lucide-react'
import { getRequirements, getClusters, getEnrichment, submitFeedback } from '../api/endpoints.js'
import { useApi } from '../hooks/useApi.js'
import { useSession } from '../state/SessionContext.jsx'
import { getErrorMessage, isConflict } from '../utils/format.js'
import { getClusterColor } from '../utils/colors.js'
import MoveToClusterModal from '../components/MoveToClusterModal.jsx'

const PAGE_SIZE = 50
const EMPTY = []

function MembershipDot({ prob }) {
  const pct = (prob ?? 0) * 100
  const color = pct >= 80 ? '#10b981' : pct >= 50 ? '#f59e0b' : '#ef4444'
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-16 h-1 bg-gray-800 rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="text-xs font-mono text-gray-500">{pct.toFixed(0)}%</span>
    </div>
  )
}

export default function RequirementsPage() {
  const { sessionId, session, reloadSession } = useSession()
  const navigate = useNavigate()

  const [search, setSearch] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [filterCluster, setFilterCluster] = useState('all')
  const [filterNoise, setFilterNoise] = useState('all') // all | noise | clustered
  const [sortField, setSortField] = useState('req_id')
  const [sortDir, setSortDir] = useState('asc')
  const [page, setPage] = useState(1)
  const [expandedEnrichment, setExpandedEnrichment] = useState({})
  const [moveError, setMoveError] = useState(null)
  const [conflict, setConflict] = useState(false)

  const [selectedRequirement, setSelectedRequirement] = useState(null)
  const [isModalOpen, setIsModalOpen] = useState(false)

  // Debounce search input
  useEffect(() => {
    const handler = setTimeout(() => setDebouncedSearch(search), 300)
    return () => clearTimeout(handler)
  }, [search])

  const { data: clustersData, reload: reloadClusters } = useApi(
    (o) => (sessionId ? getClusters(sessionId, o) : Promise.resolve([])), [sessionId])
  const { data: enrichmentRows } = useApi(
    (o) => (sessionId ? getEnrichment(sessionId, o).catch(() => []) : Promise.resolve([])), [sessionId])
  const clusters = clustersData ?? EMPTY

  const query = { page, page_size: PAGE_SIZE, sort: sortField, dir: sortDir }
  if (filterCluster !== 'all' && filterCluster !== '-1') query.cluster_id = filterCluster
  if (filterCluster === '-1' || filterNoise === 'noise') query.noise = true
  else if (filterNoise === 'clustered') query.noise = false
  if (debouncedSearch) query.search = debouncedSearch

  const { data: reqPage, error, loading: requirementsLoading, reload } = useApi(
    (o) => (sessionId ? getRequirements(sessionId, query, o) : Promise.resolve({ items: [], total: 0 })),
    [sessionId, page, filterCluster, filterNoise, sortField, sortDir, debouncedSearch])
  const requirements = reqPage?.items || []
  const totalMatched = reqPage?.total || 0

  const refresh = () => {
    setConflict(false); setMoveError(null)
    reload(); reloadClusters(); reloadSession()
  }

  const handleReassignSubmit = async (payload) => {
    if (!selectedRequirement) return
    try {
      await submitFeedback(sessionId, { requirement_id: selectedRequirement.id, ...payload })
      setIsModalOpen(false)
      setSelectedRequirement(null)
      refresh()
    } catch (err) {
      setIsModalOpen(false)
      if (isConflict(err)) setConflict(true)
      else setMoveError(getErrorMessage(err, 'Failed to reassign cluster.'))
    }
  }

  const clusterLabelMap = useMemo(() => {
    const m = {}
    clusters.forEach(c => { m[c.cluster_id] = c.label })
    return m
  }, [clusters])

  const enrichmentMap = useMemo(() => {
    const map = {}
    ;(enrichmentRows || []).forEach(row => { map[row.req_id] = row })
    return map
  }, [enrichmentRows])

  const totalPages = Math.max(1, Math.ceil(totalMatched / PAGE_SIZE))
  const paginated = requirements

  const handleSort = (field) => {
    if (sortField === field) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortField(field); setSortDir('asc') }
    setPage(1)
  }

  const renderSortHeader = (field, label) => (
    <button onClick={() => handleSort(field)}
      className="flex items-center gap-1 hover:text-gray-200 transition-colors group">
      {label}
      <ArrowUpDown size={11} className={`${sortField === field ? 'text-brand-400' : 'text-gray-700 group-hover:text-gray-500'}`} />
    </button>
  )

  if (!sessionId) return (
    <div className="p-8 text-gray-400">
      <Link to="/" className="text-brand-400 hover:underline">Select or upload a session first</Link>
    </div>
  )
  if (error) return (
    <div className="p-8 text-red-400">
      {getErrorMessage(error, 'Failed to load requirements.')}{' '}
      <button onClick={reload} className="btn-secondary text-xs py-1 px-3">Retry</button>
    </div>
  )

  return (
    <div className="p-6 flex flex-col gap-4 h-screen">
      {/* Header */}
      <div className="flex-shrink-0">
        <h1 className="text-xl font-bold text-white">Requirements</h1>
        <p className="text-sm text-gray-400 mt-0.5">
          {totalMatched} of {session?.total_requirements ?? 0} requirements
        </p>
      </div>

      {(conflict || moveError) && (
        <div className="flex-shrink-0 card p-3 border-amber-900/30 flex items-center gap-3">
          <AlertTriangle size={14} className="text-amber-400 flex-shrink-0" />
          <p className="text-sm text-amber-300/80 flex-1">
            {conflict ? 'Data changed (another edit or a running clustering job). Refresh and try again.' : moveError}
          </p>
          {conflict && <button onClick={refresh} className="btn-secondary text-xs py-1 px-3">Refresh</button>}
        </div>
      )}

      {/* Filters */}
      <div className="flex gap-3 flex-shrink-0 flex-wrap">
        <div className="relative flex-1 min-w-48">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
          <input
            type="text"
            placeholder="Search ID, text, module…"
            value={search}
            onChange={e => { setSearch(e.target.value); setPage(1) }}
            className="input text-sm pl-9 w-full"
          />
        </div>

        <select value={filterCluster} onChange={e => { setFilterCluster(e.target.value); setPage(1) }}
          className="input text-sm">
          <option value="all">All Clusters</option>
          <option value="-1">Noise</option>
          {clusters.map(c => (
            <option key={c.cluster_id} value={String(c.cluster_id)}>
              {c.label} ({c.size})
            </option>
          ))}
        </select>

        <select value={filterNoise} onChange={e => { setFilterNoise(e.target.value); setPage(1) }}
          className="input text-sm">
          <option value="all">All Types</option>
          <option value="clustered">Clustered only</option>
          <option value="noise">Noise only</option>
        </select>
      </div>

      {/* Table */}
      <div className="card flex-1 min-h-0 flex flex-col overflow-hidden">
        <div className="overflow-auto flex-1">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-gray-900 border-b border-gray-800 z-10">
              <tr className="text-xs font-medium text-gray-500 uppercase tracking-wider">
                <th className="text-left px-4 py-3 w-28">
                  {renderSortHeader('req_id', 'ID')}
                </th>
                <th className="text-left px-4 py-3">
                  {renderSortHeader('text', 'Requirement Text')}
                </th>
                <th className="text-left px-4 py-3 w-48">
                  {renderSortHeader('cluster_id', 'Cluster')}
                </th>
                <th className="text-left px-4 py-3 w-32">
                  {renderSortHeader('probability', 'Membership')}
                </th>
                <th className="w-40 px-4 py-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800/50">
              {requirementsLoading ? (
                <tr>
                  <td colSpan={5} className="text-center py-12">
                    <div className="flex items-center justify-center gap-2 text-gray-500">
                      <Loader size={16} className="animate-spin text-brand-400" />
                      <span>Loading requirements...</span>
                    </div>
                  </td>
                </tr>
              ) : paginated.length === 0 ? (
                <tr>
                  <td colSpan={5} className="text-center py-12 text-gray-500">
                    No requirements match your filters.
                  </td>
                </tr>
              ) : paginated.map(r => {
                const color = getClusterColor(r.cluster_id)
                const clusterLabel = r.is_noise || r.cluster_id == null
                  ? 'Noise'
                  : (clusterLabelMap[r.cluster_id] || `Cluster ${r.cluster_id}`)
                const enrichment = enrichmentMap[r.req_id]
                const enrichmentOpen = !!expandedEnrichment[r.req_id]
                const warningCount = enrichment?.warnings?.length || 0
                return (
                  <tr key={r.id}
                    className="hover:bg-gray-800/40 transition-colors group cursor-pointer"
                    onClick={() => !r.is_noise && r.cluster_id != null && navigate(`/s/${sessionId}/clusters/${r.cluster_id}`)}>
                    <td className="px-4 py-3">
                      <span className="font-mono text-xs text-gray-400">{r.req_id}</span>
                    </td>
                    <td className="px-4 py-3">
                      <p className="text-gray-200 leading-relaxed line-clamp-2">{r.text}</p>
                      {enrichment && (
                        <div className="mt-2">
                          <button
                            type="button"
                            onClick={(event) => {
                              event.stopPropagation()
                              setExpandedEnrichment(prev => ({ ...prev, [r.req_id]: !prev[r.req_id] }))
                            }}
                            className="inline-flex items-center gap-1.5 text-xs text-brand-300 hover:text-brand-200"
                          >
                            <Sparkles size={12} />
                            Enriched available
                            <span className={`badge ${enrichment.confidence >= 0.7 ? 'bg-emerald-500/15 text-emerald-300' : 'bg-amber-500/15 text-amber-300'}`}>
                              {enrichment.confidence === null || enrichment.confidence === undefined ? '-' : enrichment.confidence.toFixed(2)}
                            </span>
                            {warningCount > 0 && (
                              <span className="badge bg-amber-500/15 text-amber-300">
                                {warningCount} warnings
                              </span>
                            )}
                            <ChevronDown size={12} className={`transition-transform ${enrichmentOpen ? 'rotate-180' : ''}`} />
                          </button>
                          {enrichmentOpen && (
                            <div
                              className="mt-2 rounded-lg border border-gray-800 bg-gray-950/50 p-3"
                              onClick={event => event.stopPropagation()}
                            >
                              <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-1">Expanded Text</div>
                              <p className="text-sm text-gray-300 leading-relaxed">{enrichment.expanded_text || '-'}</p>
                              {enrichment.domain_terms?.length > 0 && (
                                <div className="flex flex-wrap gap-1.5 mt-2">
                                  {enrichment.domain_terms.slice(0, 8).map(term => (
                                    <span key={term} className="badge bg-brand-600/15 text-brand-300 border border-brand-500/20">{term}</span>
                                  ))}
                                </div>
                              )}
                              {warningCount > 0 && (
                                <div className="mt-2 space-y-1">
                                  {enrichment.warnings.slice(0, 3).map(warning => (
                                    <div key={warning} className="flex items-start gap-1.5 text-xs text-amber-300">
                                      <AlertTriangle size={11} className="mt-0.5 flex-shrink-0" />
                                      <span>{warning}</span>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full flex-shrink-0"
                          style={{ backgroundColor: color }} />
                        <span className="text-xs text-gray-300 truncate max-w-32">{clusterLabel}</span>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      {r.is_noise
                        ? <span className="text-xs text-gray-600">-</span>
                        : <MembershipDot prob={r.probability} />
                      }
                    </td>
                    <td className="px-4 py-3 text-right" onClick={(e) => e.stopPropagation()}>
                      <div className="flex items-center justify-end gap-2">
                        <button
                          type="button"
                          onClick={() => {
                            setSelectedRequirement(r)
                            setIsModalOpen(true)
                          }}
                          className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 text-xs transition-colors"
                        >
                          Reassign
                        </button>
                        <button
                          type="button"
                          onClick={() => navigate(`/s/${sessionId}/clusters/${r.cluster_id}`)}
                          className={`p-1 rounded hover:bg-slate-800 text-slate-500 hover:text-slate-300 transition-colors ${r.is_noise ? 'invisible' : ''}`}
                        >
                          <ChevronRight size={14} />
                        </button>
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between px-4 py-3 border-t border-gray-800 flex-shrink-0">
            <span className="text-xs text-gray-500">
              Page {page} of {totalPages} · {totalMatched} results
            </span>
            <div className="flex gap-2">
              <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}
                className="btn-secondary text-xs py-1 px-3 disabled:opacity-40">← Prev</button>
              <button onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page === totalPages}
                className="btn-secondary text-xs py-1 px-3 disabled:opacity-40">Next →</button>
            </div>
          </div>
        )}
      </div>

      <MoveToClusterModal
        isOpen={isModalOpen}
        onClose={() => {
          setIsModalOpen(false)
          setSelectedRequirement(null)
        }}
        requirement={selectedRequirement}
        clusters={clusters}
        onSubmit={handleReassignSubmit}
      />
    </div>
  )
}
