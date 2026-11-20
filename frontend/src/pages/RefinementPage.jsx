import { useState, useMemo } from 'react'
import { Link } from 'react-router-dom'
import { Wrench, GitMerge, Scissors, Clock, AlertTriangle, RefreshCw } from 'lucide-react'
import { startSuggestions, getSuggestions, decideSuggestion, getAudit, getClusters, getCoherence, getMetrics } from '../api/endpoints.js'
import { useApi } from '../hooks/useApi.js'
import { useJob } from '../hooks/useJob.js'
import { useSession } from '../state/SessionContext.jsx'
import { getErrorMessage, formatTimestamp, isConflict } from '../utils/format.js'
import SuggestionCard from '../components/SuggestionCard.jsx'
import JobProgress from '../components/JobProgress.jsx'

export default function RefinementPage() {
  const { sessionId } = useSession()
  const [activeTab, setActiveTab] = useState('merge')
  const [jobId, setJobId] = useState(null)
  const [applyingId, setApplyingId] = useState(null)
  const [error, setError] = useState(null)
  const [conflict, setConflict] = useState(false)

  const sugg = useApi((o) => (sessionId ? getSuggestions(sessionId, undefined, o) : Promise.resolve([])), [sessionId])
  const auditApi = useApi((o) => (sessionId ? getAudit(sessionId, o) : Promise.resolve([])), [sessionId])
  const clustersApi = useApi((o) => (sessionId ? getClusters(sessionId, o) : Promise.resolve([])), [sessionId])

  const cohApi = useApi((o) => (sessionId ? getCoherence(sessionId, o) : Promise.resolve([])), [sessionId, sugg.data])
  const coherenceScores = cohApi.data || []
  const metricsApi = useApi((o) => (sessionId ? getMetrics(sessionId, o) : Promise.resolve(null)), [sessionId, sugg.data])
  const analyzed = metricsApi.data?.refinement_report

  const all = sugg.data || []
  const mergeSuggestions = all.filter(s => s.kind === 'merge')
  const splitSuggestions = all.filter(s => s.kind === 'split')
  const auditLog = auditApi.data || []
  const clusterInfo = useMemo(() => {
    const info = {}
    ;(clustersApi.data || []).forEach(c => {
      info[c.cluster_id] = { label: c.label, keywords: c.keywords || [], size: c.size }
    })
    return info
  }, [clustersApi.data])

  const refresh = () => {
    setConflict(false)
    setError(null)
    sugg.reload()
    auditApi.reload()
    clustersApi.reload()
  }

  const { job, cancel } = useJob(jobId, (j) => {
    setJobId(null)
    if (j.status === 'failed') setError(j.error || 'Failed to generate suggestions.')
    refresh()
  })
  const generating = jobId != null

  const handleGenerate = async () => {
    setError(null)
    try {
      const j = await startSuggestions(sessionId, {})
      setJobId(j.id)
    } catch (err) {
      setError(getErrorMessage(err, 'Failed to generate suggestions.'))
    }
  }

  const decide = (decision) => async (suggestionId) => {
    setApplyingId(suggestionId)
    setError(null)
    setConflict(false)
    try {
      await decideSuggestion(sessionId, suggestionId, decision)
      refresh()
    } catch (err) {
      if (isConflict(err)) setConflict(true)
      else setError(getErrorMessage(err, `Failed to ${decision} suggestion.`))
    } finally {
      setApplyingId(null)
    }
  }
  const handleAccept = decide('accept')
  const handleReject = decide('reject')

  if (!sessionId) {
    return (
      <div className="p-8">
        <Link to="/" className="text-brand-400 hover:underline">Select or upload a session first</Link>
      </div>
    )
  }

  const currentSuggestions = activeTab === 'merge' ? mergeSuggestions : activeTab === 'split' ? splitSuggestions : []
  const pendingCount = [...mergeSuggestions, ...splitSuggestions].filter(s => s.status === 'pending').length

  return (
    <div className="p-8 max-w-6xl">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-1">
          <Wrench size={20} className="text-brand-400" />
          <h1 className="text-2xl font-bold text-white">Cluster Refinement</h1>
        </div>
        <p className="text-gray-400 text-sm">
          Analyze cluster quality and review merge/split suggestions powered by ClusterLLM techniques.
        </p>
      </div>

      <div className="card p-4 mb-6 flex items-center justify-between gap-4 flex-wrap">
        <span className="text-sm text-gray-400">Session #{sessionId}</span>
        <button
          onClick={handleGenerate}
          disabled={generating}
          className="btn-primary flex items-center gap-2 text-sm px-5 py-2.5 disabled:opacity-50"
        >
          <RefreshCw size={14} />
          Analyze Clusters
        </button>
      </div>

      {generating && <div className="mb-4"><JobProgress job={job} onCancel={cancel} title="Analyzing" /></div>}

      {conflict && (
        <div className="card p-3 mb-4 border-amber-900/30 bg-amber-950/10 flex items-center gap-2">
          <AlertTriangle size={14} className="text-amber-400 flex-shrink-0" />
          <p className="text-sm text-amber-300/80 flex-1">Data changed. Refresh and try again.</p>
          <button onClick={refresh} className="btn-secondary text-sm">Refresh</button>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="card p-3 mb-4 border-amber-900/30 bg-amber-950/10 flex items-start gap-2">
          <AlertTriangle size={14} className="text-amber-400 mt-0.5 flex-shrink-0" />
          <p className="text-sm text-amber-300/80">{error}</p>
        </div>
      )}

      {coherenceScores.length > 0 && (
        <div className="card p-4 mb-6">
          <h2 className="text-sm font-semibold text-gray-300 mb-3">Cluster Coherence Scores</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
            {coherenceScores.map(cs => (
              <div key={cs.cluster_id} title={cs.assessment} className="bg-gray-800/40 rounded-lg p-3 border border-gray-700/30">
                <div className="text-xs text-gray-500 truncate mb-1">
                  {cs.top_keywords?.slice(0, 2).join(', ') || `Cluster ${cs.cluster_id}`}
                </div>
                <div className={`text-lg font-bold ${
                  ({ high: 'text-emerald-400', moderate: 'text-amber-400' })[cs.level] || 'text-red-400'
                }`}>
                  {cs.coherence_score.toFixed(3)}
                </div>
                <div className="text-xs text-gray-500">{cs.size} reqs - {cs.assessment}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 mb-4 bg-gray-900 rounded-lg p-1 w-fit">
        {[
          { key: 'merge', icon: GitMerge, label: 'Merge', count: mergeSuggestions.length },
          { key: 'split', icon: Scissors, label: 'Split', count: splitSuggestions.length },
          { key: 'audit', icon: Clock, label: 'Audit Log', count: auditLog.length },
        ].map(tab => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm transition-colors ${
              activeTab === tab.key
                ? 'bg-gray-800 text-white font-medium'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            <tab.icon size={14} />
            {tab.label}
            {tab.count > 0 && (
              <span className="text-xs bg-gray-700 text-gray-300 px-1.5 py-0.5 rounded-full ml-1">
                {tab.count}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Suggestions */}
      {activeTab !== 'audit' && (
        <div className="space-y-3">
          {currentSuggestions.length === 0 ? (
            <div className="card p-8 text-center text-gray-500 text-sm">
              {generating ? 'Analyzing clusters...'
                : analyzed ? `Last analysis found no ${activeTab} that improves cluster separation.`
                  : 'No suggestions yet. Click "Analyze Clusters" to generate.'}
            </div>
          ) : (
            currentSuggestions.map(s => (
              <SuggestionCard
                key={s.id}
                suggestion={s}
                clusterInfo={clusterInfo}
                onAccept={handleAccept}
                onReject={handleReject}
                isApplying={applyingId === s.id}
              />
            ))
          )}
        </div>
      )}

      {/* Audit log */}
      {activeTab === 'audit' && (
        <div className="card overflow-hidden">
          {auditLog.length === 0 ? (
            <div className="p-8 text-center text-gray-500 text-sm">
              No refinements applied yet.
            </div>
          ) : (
            <div className="divide-y divide-gray-800/50">
              {auditLog.map(entry => (
                <div key={entry.id} className="px-4 py-3 flex items-center gap-3">
                  <div className={`w-2 h-2 rounded-full ${
                    entry.action.includes('accept') || entry.action === 'applied' ? 'bg-emerald-400'
                      : entry.action.includes('reject') ? 'bg-red-400'
                        : 'bg-gray-500'
                  }`} />
                  <div className="flex-1">
                    <div className="text-sm text-gray-200">
                      {entry.details?.suggestion_id != null ? `Suggestion #${entry.details.suggestion_id} - ` : ''}<span className="font-medium capitalize">{entry.action}</span>
                    </div>
                    <div className="text-xs text-gray-500">
                      {entry.created_at ? formatTimestamp(entry.created_at) : 'Unknown time'}{entry.actor ? ` · ${entry.actor}` : ''}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Pending count */}
      {pendingCount > 0 && (
        <div className="mt-4 text-xs text-gray-500 text-center">
          {pendingCount} suggestion{pendingCount !== 1 ? 's' : ''} pending review
        </div>
      )}
    </div>
  )
}
