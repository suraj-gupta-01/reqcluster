import React, { useState, useEffect, useMemo } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { Loader, Search, Filter, ChevronRight, ArrowUpDown } from 'lucide-react'
import { getRequirements, getClusters } from '../utils/api.js'
import { getClusterColor } from '../utils/colors.js'

const PAGE_SIZE = 50

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
  const { sessionId } = useParams()
  const navigate = useNavigate()

  const [requirements, setRequirements] = useState([])
  const [clusters, setClusters] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const [search, setSearch] = useState('')
  const [filterCluster, setFilterCluster] = useState('all')
  const [filterNoise, setFilterNoise] = useState('all') // all | noise | clustered
  const [sortField, setSortField] = useState('req_id')
  const [sortDir, setSortDir] = useState('asc')
  const [page, setPage] = useState(1)

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const [reqs, clus] = await Promise.all([
          getRequirements(parseInt(sessionId)),
          getClusters(parseInt(sessionId)),
        ])
        if (!cancelled) { setRequirements(reqs); setClusters(clus) }
      } catch {
        if (!cancelled) setError('Failed to load requirements.')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    load()
    return () => { cancelled = true }
  }, [sessionId])

  const clusterLabelMap = useMemo(() => {
    const m = {}
    clusters.forEach(c => { m[c.cluster_id] = c.label })
    return m
  }, [clusters])

  const filtered = useMemo(() => {
    let list = requirements

    if (search) {
      const q = search.toLowerCase()
      list = list.filter(r =>
        r.text?.toLowerCase().includes(q) ||
        r.req_id?.toLowerCase().includes(q) ||
        r.module?.toLowerCase().includes(q) ||
        r.section?.toLowerCase().includes(q)
      )
    }

    if (filterCluster !== 'all') {
      list = list.filter(r => String(r.cluster_id) === filterCluster)
    }

    if (filterNoise === 'noise') list = list.filter(r => r.is_noise)
    else if (filterNoise === 'clustered') list = list.filter(r => !r.is_noise)

    list = [...list].sort((a, b) => {
      let av, bv
      if (sortField === 'req_id') { av = a.req_id || ''; bv = b.req_id || '' }
      else if (sortField === 'cluster_id') { av = a.cluster_id ?? 999; bv = b.cluster_id ?? 999 }
      else if (sortField === 'membership_prob') { av = a.membership_prob ?? 0; bv = b.membership_prob ?? 0 }
      else if (sortField === 'module') { av = a.module || ''; bv = b.module || '' }
      else { av = a.text || ''; bv = b.text || '' }
      if (av < bv) return sortDir === 'asc' ? -1 : 1
      if (av > bv) return sortDir === 'asc' ? 1 : -1
      return 0
    })

    return list
  }, [requirements, search, filterCluster, filterNoise, sortField, sortDir])

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
  const paginated = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)

  const handleSort = (field) => {
    if (sortField === field) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortField(field); setSortDir('asc') }
    setPage(1)
  }

  const SortHeader = ({ field, children }) => (
    <button onClick={() => handleSort(field)}
      className="flex items-center gap-1 hover:text-gray-200 transition-colors group">
      {children}
      <ArrowUpDown size={11} className={`${sortField === field ? 'text-brand-400' : 'text-gray-700 group-hover:text-gray-500'}`} />
    </button>
  )

  if (loading) return (
    <div className="flex items-center justify-center h-64">
      <Loader size={24} className="animate-spin text-brand-400" />
    </div>
  )
  if (error) return <div className="p-8 text-red-400">{error}</div>

  return (
    <div className="p-6 flex flex-col gap-4 h-screen">
      {/* Header */}
      <div className="flex-shrink-0">
        <h1 className="text-xl font-bold text-white">Requirements</h1>
        <p className="text-sm text-gray-400 mt-0.5">
          {filtered.length} of {requirements.length} requirements
        </p>
      </div>

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
                  <SortHeader field="req_id">ID</SortHeader>
                </th>
                <th className="text-left px-4 py-3">
                  <SortHeader field="text">Requirement Text</SortHeader>
                </th>
                <th className="text-left px-4 py-3 w-32">
                  <SortHeader field="module">Module</SortHeader>
                </th>
                <th className="text-left px-4 py-3 w-48">
                  <SortHeader field="cluster_id">Cluster</SortHeader>
                </th>
                <th className="text-left px-4 py-3 w-32">
                  <SortHeader field="membership_prob">Membership</SortHeader>
                </th>
                <th className="w-8 px-4 py-3" />
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800/50">
              {paginated.length === 0 ? (
                <tr>
                  <td colSpan={6} className="text-center py-12 text-gray-500">
                    No requirements match your filters.
                  </td>
                </tr>
              ) : paginated.map(r => {
                const color = getClusterColor(r.cluster_id)
                const clusterLabel = r.is_noise
                  ? 'Noise'
                  : (clusterLabelMap[r.cluster_id] || `Cluster ${r.cluster_id}`)
                return (
                  <tr key={r.id}
                    className="hover:bg-gray-800/40 transition-colors group cursor-pointer"
                    onClick={() => !r.is_noise && navigate(`/cluster/${sessionId}/${r.cluster_id}`)}>
                    <td className="px-4 py-3">
                      <span className="font-mono text-xs text-gray-400">{r.req_id}</span>
                    </td>
                    <td className="px-4 py-3">
                      <p className="text-gray-200 leading-relaxed line-clamp-2">{r.text}</p>
                      {r.section && (
                        <span className="text-xs text-gray-600 mt-0.5 block">{r.section}</span>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      {r.module ? (
                        <span className="badge bg-gray-800 text-gray-400">{r.module}</span>
                      ) : (
                        <span className="text-gray-700">—</span>
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
                        ? <span className="text-xs text-gray-600">—</span>
                        : <MembershipDot prob={r.membership_prob} />
                      }
                    </td>
                    <td className="px-4 py-3">
                      {!r.is_noise && (
                        <ChevronRight size={14} className="text-gray-700 group-hover:text-gray-400 transition-colors" />
                      )}
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
              Page {page} of {totalPages} · {filtered.length} results
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
    </div>
  )
}
