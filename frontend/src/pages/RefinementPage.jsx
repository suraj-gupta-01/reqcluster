import { useState, useEffect, useCallback } from 'react'
import { Wrench, Loader, GitMerge, Scissors, Clock, AlertTriangle, RefreshCw } from 'lucide-react'
import {
  getSessions,
  generateSuggestions,
  getSuggestions,
  applySuggestion,
  getAuditLog,
  getClusters,
  getErrorMessage,
} from '../utils/api.js'
import SuggestionCard from '../components/SuggestionCard.jsx'

export default function RefinementPage() {
  const [sessions, setSessions] = useState([])
  const [selectedSession, setSelectedSession] = useState(null)
  const [activeTab, setActiveTab] = useState('merge')

  const [mergeSuggestions, setMergeSuggestions] = useState([])
  const [splitSuggestions, setSplitSuggestions] = useState([])
  const [coherenceScores, setCoherenceScores] = useState([])
  const [clusterSummaries, setClusterSummaries] = useState([])
  const [auditLog, setAuditLog] = useState([])
  const [clusterInfo, setClusterInfo] = useState({})

  const [generating, setGenerating] = useState(false)
  const [applyingId, setApplyingId] = useState(null)
  const [error, setError] = useState(null)
  const [loadingSessions, setLoadingSessions] = useState(true)

  // Load sessions
  useEffect(() => {
    getSessions()
      .then(list => {
        const doneSessions = list.filter(s => s.status === 'done')
        setSessions(doneSessions)
        if (doneSessions.length > 0 && !selectedSession) {
          setSelectedSession(doneSessions[0].id)
        }
      })
      .catch(() => setError('Failed to load sessions.'))
      .finally(() => setLoadingSessions(false))
  }, [])

  // Load existing suggestions when session changes
  const loadSuggestions = useCallback(async (sessionId) => {
    try {
      const [all, audit, clusters] = await Promise.all([
        getSuggestions(sessionId),
        getAuditLog(sessionId),
        getClusters(sessionId),
      ])
      setMergeSuggestions(all.filter(s => s.suggestion_type === 'merge'))
      setSplitSuggestions(all.filter(s => s.suggestion_type === 'split'))
      setAuditLog(audit)

      const info = {}
      clusters.forEach(c => {
        info[c.cluster_id] = { label: c.label, keywords: c.keywords || [], size: c.size }
      })
      setClusterInfo(info)
      setError(null)
    } catch {
      // No suggestions yet — that's fine
    }
  }, [])

  useEffect(() => {
    if (selectedSession) loadSuggestions(selectedSession)
  }, [selectedSession, loadSuggestions])

  // Generate suggestions
  const handleGenerate = async () => {
    if (!selectedSession) return
    setGenerating(true)
    setError(null)
    try {
      const result = await generateSuggestions({ session_id: selectedSession })
      setMergeSuggestions(result.merge_suggestions || [])
      setSplitSuggestions(result.split_suggestions || [])
      setCoherenceScores(result.coherence_scores || [])
      setClusterSummaries(result.cluster_summaries || [])
      if (result.warnings?.length) {
        setError(result.warnings.join(' '))
      }
    } catch (err) {
      setError(getErrorMessage(err, 'Failed to generate suggestions.'))
    } finally {
      setGenerating(false)
    }
  }

  // Apply suggestion
  const handleAccept = async (suggestionId) => {
    setApplyingId(suggestionId)
    setError(null)
    try {
      await applySuggestion({
        session_id: selectedSession,
        suggestion_id: suggestionId,
        action: 'accept',
      })
      await loadSuggestions(selectedSession)
    } catch (err) {
      setError(getErrorMessage(err, 'Failed to apply suggestion.'))
    } finally {
      setApplyingId(null)
    }
  }

  const handleReject = async (suggestionId) => {
    setApplyingId(suggestionId)
    setError(null)
    try {
      await applySuggestion({
        session_id: selectedSession,
        suggestion_id: suggestionId,
        action: 'reject',
      })
      await loadSuggestions(selectedSession)
    } catch (err) {
      setError(getErrorMessage(err, 'Failed to reject suggestion.'))
    } finally {
      setApplyingId(null)
    }
  }

  const currentSuggestions = activeTab === 'merge' ? mergeSuggestions : activeTab === 'split' ? splitSuggestions : []
  const pendingCount = [...mergeSuggestions, ...splitSuggestions].filter(s => s.status === 'pending').length

  return (
    <div className="p-8 max-w-5xl">
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

      {/* Session selector */}
      <div className="card p-4 mb-6">
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex-1 min-w-[200px]">
            <label className="text-xs text-gray-500 block mb-1">Session</label>
            {loadingSessions ? (
              <div className="flex items-center gap-2 text-gray-400 text-sm">
                <Loader size={14} className="animate-spin" /> Loading sessions...
              </div>
            ) : sessions.length === 0 ? (
              <div className="text-sm text-gray-500">No completed sessions found. Run clustering first.</div>
            ) : (
              <select
                value={selectedSession || ''}
                onChange={e => setSelectedSession(parseInt(e.target.value))}
                className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 w-full focus:outline-none focus:ring-1 focus:ring-brand-500"
              >
                {sessions.map(s => (
                  <option key={s.id} value={s.id}>
                    #{s.id} — {s.filename} ({s.total_requirements} reqs, {s.total_clusters} clusters)
                  </option>
                ))}
              </select>
            )}
          </div>

          <button
            onClick={handleGenerate}
            disabled={!selectedSession || generating}
            className="btn-primary flex items-center gap-2 text-sm px-5 py-2.5 disabled:opacity-50 mt-4 sm:mt-0"
          >
            {generating ? (
              <>
                <Loader size={14} className="animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <RefreshCw size={14} />
                Analyze Clusters
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="card p-3 mb-4 border-amber-900/30 bg-amber-950/10 flex items-start gap-2">
          <AlertTriangle size={14} className="text-amber-400 mt-0.5 flex-shrink-0" />
          <p className="text-sm text-amber-300/80">{error}</p>
        </div>
      )}

      {/* Coherence summary */}
      {coherenceScores.length > 0 && (
        <div className="card p-4 mb-6">
          <h2 className="text-sm font-semibold text-gray-300 mb-3">Cluster Coherence Scores</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
            {coherenceScores.map(cs => (
              <div key={cs.cluster_id} className="bg-gray-800/40 rounded-lg p-3 border border-gray-700/30">
                <div className="text-xs text-gray-500 truncate mb-1">
                  {cs.top_keywords?.slice(0, 2).join(', ') || `Cluster ${cs.cluster_id}`}
                </div>
                <div className={`text-lg font-bold ${
                  cs.coherence_score >= 0.8 ? 'text-emerald-400'
                    : cs.coherence_score >= 0.6 ? 'text-amber-400'
                      : 'text-red-400'
                }`}>
                  {cs.coherence_score.toFixed(3)}
                </div>
                <div className="text-xs text-gray-500">{cs.size} reqs</div>
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
              {generating ? 'Analyzing clusters...' : 'No suggestions yet. Click "Analyze Clusters" to generate.'}
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
                    entry.action === 'applied' ? 'bg-emerald-400'
                      : entry.action === 'rejected' ? 'bg-red-400'
                        : 'bg-gray-500'
                  }`} />
                  <div className="flex-1">
                    <div className="text-sm text-gray-200">
                      Suggestion #{entry.suggestion_id} — <span className="font-medium capitalize">{entry.action}</span>
                    </div>
                    <div className="text-xs text-gray-500">
                      {entry.created_at ? new Date(entry.created_at).toLocaleString() : 'Unknown time'}
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
