import { useNavigate, Link } from 'react-router-dom'
import { Layers, Hash, AlertTriangle, ChevronRight, Loader, Tag, Sparkles, Wrench } from 'lucide-react'
import { getClusters, getEnrichmentSummary, getSuggestions } from '../api/endpoints.js'
import { useApi } from '../hooks/useApi.js'
import { useSession } from '../state/SessionContext.jsx'
import { getErrorMessage, conflictAs } from '../utils/format.js'
import SessionGate from '../components/SessionGate.jsx'
import { getClusterColor } from '../utils/colors.js'

const PROVIDER_LABELS = {
  mock: 'Mock',
  local: 'Local LLM',
  openai_compatible: 'Cloud LLM',
  'openai-compatible': 'Cloud LLM',
}

function StatCard({ icon: Icon, label, value, color = 'text-white' }) {
  return (
    <div className="stat-card">
      <div className="flex items-center gap-2 text-gray-400 mb-2">
        <Icon size={15} />
        <span className="text-xs font-medium uppercase tracking-wider">{label}</span>
      </div>
      <div className={`text-3xl font-bold ${color}`}>{value}</div>
    </div>
  )
}

function ClusterRow({ cluster, sessionId, maxSize }) {
  const navigate = useNavigate()
  const color = getClusterColor(cluster.cluster_id)

  return (
    <div
      onClick={() => navigate(`/s/${sessionId}/clusters/${cluster.cluster_id}`)}
      className="flex items-center gap-4 px-4 py-3 rounded-lg hover:bg-gray-800 cursor-pointer transition-colors group"
    >
      {/* Color dot */}
      <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />

      {/* Cluster label */}
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium text-gray-200 group-hover:text-white truncate">
          {cluster.label}
        </div>
        {cluster.keywords?.length > 0 && (
          <div className="flex gap-1 mt-1 flex-wrap">
            {cluster.keywords.slice(0, 3).map(kw => (
              <span key={kw} className="text-xs bg-gray-800 text-gray-400 px-1.5 py-0.5 rounded">
                {kw}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Bar + size */}
      <div className="flex items-center gap-3 w-32">
        <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full"
            style={{
              width: `${Math.min(100, (cluster.size / Math.max(1, maxSize)) * 100)}%`,
              backgroundColor: color,
            }}
          />
        </div>
        <span className="text-sm font-mono text-gray-400 w-6 text-right">{cluster.size}</span>
      </div>

      <ChevronRight size={14} className="text-gray-600 group-hover:text-gray-400" />
    </div>
  )
}

export default function OverviewPage() {
  return <SessionGate><Overview /></SessionGate>
}

function Overview() {
  const { sessionId, session } = useSession()
  const clustersQ = useApi((o) => getClusters(sessionId, o), [sessionId])
  const enrichQ = useApi((o) => getEnrichmentSummary(sessionId, o).catch(conflictAs(null)), [sessionId])
  const sugsQ = useApi((o) => getSuggestions(sessionId, undefined, o), [sessionId])
  const clusters = clustersQ.data || []
  const refinementSuggestions = sugsQ.data || []
  const qr = enrichQ.data?.quality_report
  const enrichmentStatus = qr?.total
    ? {
        total: qr.total,
        succeeded: qr.succeeded,
        warnings: qr.warnings || [],
        provider: enrichQ.data.provider,
        status: 'available',
      }
    : null

  if (clustersQ.loading && !clustersQ.data) return (
    <div className="flex items-center justify-center h-64">
      <Loader size={24} className="animate-spin text-brand-400" />
    </div>
  )

  if (clustersQ.error) return (
    <div className="p-8 text-red-400">{getErrorMessage(clustersQ.error, 'Failed to load session data.')}</div>
  )

  const noiseCount = session?.noise_count || 0
  const totalClusters = session?.total_clusters || 0
  const totalReqs = session?.total_requirements || 0

  return (
    <div className="p-8">
      {enrichQ.error && (
        <div className="mb-4 text-xs text-red-400">{getErrorMessage(enrichQ.error, 'Failed to load enrichment status.')}</div>
      )}
      {/* Header */}
      <div className="mb-6">
        <div className="text-xs text-gray-500 mb-1 font-mono">{session?.name}</div>
        <h1 className="text-2xl font-bold text-white">Cluster Overview</h1>
        <p className="text-gray-400 mt-1">
          {session?.status === 'done'
            ? 'Pipeline complete - explore your requirement clusters below.'
            : `Status: ${session?.status}`}
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4 mb-8">
        <StatCard icon={Hash} label="Requirements" value={totalReqs} />
        <StatCard icon={Layers} label="Clusters" value={totalClusters} color="text-brand-400" />
        <StatCard icon={AlertTriangle} label="Noise Points" value={noiseCount} color="text-amber-400" />
        <StatCard
          icon={Tag}
          label="Coverage"
          value={totalReqs > 0 ? `${(((totalReqs - noiseCount) / totalReqs) * 100).toFixed(1)}%` : '-'}
          color="text-emerald-400"
        />
      </div>

      {/* Quick links */}
      <div className="flex gap-3 mb-8">
        <Link to={`/s/${sessionId}/scatter`} className="btn-secondary flex items-center gap-2 text-sm">
          <span>Scatter Plot</span>
          <ChevronRight size={14} />
        </Link>
        <Link to={`/s/${sessionId}/graph`} className="btn-secondary flex items-center gap-2 text-sm">
          <span>Similarity Graph</span>
          <ChevronRight size={14} />
        </Link>
        <Link to={`/s/${sessionId}/requirements`} className="btn-secondary flex items-center gap-2 text-sm">
          <span>All Requirements</span>
          <ChevronRight size={14} />
        </Link>
        <Link to={`/s/${sessionId}/enrichment`} className="btn-secondary flex items-center gap-2 text-sm">
          <Sparkles size={14} />
          <span>Enrichment</span>
          <ChevronRight size={14} />
        </Link>
        <Link to={`/s/${sessionId}/refinement`} className="btn-secondary flex items-center gap-2 text-sm">
          <Wrench size={14} />
          <span>Refinement</span>
          <ChevronRight size={14} />
        </Link>
      </div>

      {/* Phase 2 enrichment summary */}
      <div className="card p-4 mb-4">
        <div className="flex items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <Sparkles size={15} className="text-brand-400" />
              <h2 className="text-sm font-semibold text-gray-300">Phase 2 Enrichment</h2>
            </div>
            <p className="text-sm text-gray-500 mt-1">
              {enrichmentStatus
                ? `${enrichmentStatus.succeeded || 0} of ${enrichmentStatus.total || 0} requirements enriched with ${PROVIDER_LABELS[enrichmentStatus.provider?.toLowerCase()] || enrichmentStatus.provider || 'no provider yet'}.`
                : 'No enrichment status is available yet.'}
            </p>
          </div>
          <div className="flex gap-3 text-right">
            <div>
              <div className="text-lg font-semibold text-emerald-400">{enrichmentStatus?.succeeded ?? 0}</div>
              <div className="text-xs text-gray-500">Enriched</div>
            </div>
            <div>
              <div className="text-lg font-semibold text-amber-400">{enrichmentStatus?.warnings?.length ?? 0}</div>
              <div className="text-xs text-gray-500">Warnings</div>
            </div>
            <div>
              <div className="text-lg font-semibold text-brand-400">{enrichmentStatus?.status || 'not started'}</div>
              <div className="text-xs text-gray-500">Status</div>
            </div>
          </div>
        </div>
      </div>

      {/* Phase 3 refinement summary */}
      <div className="card p-4 mb-8">
        <div className="flex items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <Wrench size={15} className="text-orange-400" />
              <h2 className="text-sm font-semibold text-gray-300">Phase 3 Refinement</h2>
            </div>
            <p className="text-sm text-gray-500 mt-1">
              {refinementSuggestions.length > 0
                ? `${refinementSuggestions.length} suggestion${refinementSuggestions.length !== 1 ? 's' : ''} generated for cluster boundary correction.`
                : 'No refinement suggestions generated yet.'}
            </p>
          </div>
          <div className="flex gap-3 text-right">
            <div>
              <div className="text-lg font-semibold text-amber-400">
                {refinementSuggestions.filter(s => s.status === 'pending').length}
              </div>
              <div className="text-xs text-gray-500">Pending</div>
            </div>
            <div>
              <div className="text-lg font-semibold text-emerald-400">
                {refinementSuggestions.filter(s => s.status === 'accepted').length}
              </div>
              <div className="text-xs text-gray-500">Accepted</div>
            </div>
            <div>
              <div className="text-lg font-semibold text-red-400">
                {refinementSuggestions.filter(s => s.status === 'rejected').length}
              </div>
              <div className="text-xs text-gray-500">Rejected</div>
            </div>
          </div>
        </div>
      </div>

      {/* Cluster list */}
      <div className="card">
        <div className="px-4 py-3 border-b border-gray-800 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-gray-300">Clusters</h2>
          <span className="text-xs text-gray-500">{clusters.length} clusters · click to explore</span>
        </div>
        <div className="p-2 divide-y divide-gray-800/50">
          {clusters.length === 0 ? (
            <div className="text-center py-8 text-gray-500 text-sm">
              {totalClusters === 0 ? 'All requirements were classified as noise (0 clusters). ' : ''}No clusters found. Try lowering min_cluster_size on the Upload page and re-running.
            </div>
          ) : (
            clusters.map((c) => (
              <ClusterRow key={c.cluster_id} cluster={c} sessionId={sessionId} maxSize={Math.max(0, ...clusters.map(x => x.size))} />
            ))
          )}
        </div>
      </div>

      {/* Noise note */}
      {noiseCount > 0 && (
        <div className="mt-4 card p-4 border-amber-900/30 bg-amber-950/10 flex items-start gap-3">
          <AlertTriangle size={15} className="text-amber-400 mt-0.5 flex-shrink-0" />
          <p className="text-sm text-amber-300/80">
            <span className="font-medium">{noiseCount} requirements</span> were classified as noise (cluster -1).
            These requirements are too dissimilar from any cluster. Try lowering <code className="font-mono text-xs">min_cluster_size</code> to absorb them.
          </p>
        </div>
      )}
    </div>
  )
}
