import React, { useState, useEffect, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  Inbox,
  Loader,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Download,
  Users,
  Award,
  ChevronDown,
  ChevronUp,
  FileSpreadsheet,
  FileCode,
} from 'lucide-react'
import {
  getSessions,
  getFeedbackQueue,
  reviewFeedback,
  getConstraints,
  getFeedbackExportUrl,
  getErrorMessage,
  getClusters,
  getRequirements,
} from '../utils/api.js'

export default function ReviewQueuePage() {
  const { sessionId } = useParams()
  const navigate = useNavigate()
  const selectedSession = sessionId ? parseInt(sessionId, 10) : null

  const [sessions, setSessions] = useState([])
  const [loadingSessions, setLoadingSessions] = useState(true)

  const [queue, setQueue] = useState([])
  const [constraintsInfo, setConstraintsInfo] = useState({ constraint_pairs: [], conflicts: [], has_conflicts: false })
  const [clusterInfo, setClusterInfo] = useState({})
  const [reqMap, setReqMap] = useState({})
  const [expandedId, setExpandedId] = useState(null)

  const [activeTab, setActiveTab] = useState('pending') // pending, history
  const [reviewingId, setReviewingId] = useState(null)
  const [error, setError] = useState(null)
  const [exportOpen, setExportOpen] = useState(false)

  // Load done sessions
  useEffect(() => {
    getSessions()
      .then((list) => {
        const done = list.filter((s) => s.status === 'done')
        setSessions(done)
        if (done.length > 0 && !selectedSession) {
          navigate(`/review-queue/${done[0].id}`, { replace: true })
        }
      })
      .catch(() => setError('Failed to load sessions.'))
      .finally(() => setLoadingSessions(false))
  }, [selectedSession, navigate])

  // Load queue, constraints, clusters, and requirement mappings
  const loadQueueData = useCallback(async (sessionId) => {
    try {
      const [queueData, constraintsData, clustersData, reqsData] = await Promise.all([
        getFeedbackQueue(sessionId),
        getConstraints(sessionId),
        getClusters(sessionId),
        getRequirements(sessionId),
      ])

      setQueue(queueData)
      setConstraintsInfo(constraintsData)

      const cMap = {}
      clustersData.forEach((c) => {
        cMap[c.cluster_id] = c.label
      })
      setClusterInfo(cMap)

      const rMap = {}
      reqsData.forEach((r) => {
        rMap[r.id] = r
      })
      setReqMap(rMap)
      setError(null)
    } catch (err) {
      setError(getErrorMessage(err, 'Failed to load queue details.'))
    }
  }, [])

  useEffect(() => {
    if (selectedSession) {
      loadQueueData(selectedSession)
    }
  }, [selectedSession, loadQueueData])

  // Approve / Reject handlers
  const handleReview = async (feedbackId, status) => {
    if (!selectedSession) return
    setReviewingId(feedbackId)
    setError(null)
    try {
      await reviewFeedback({
        session_id: selectedSession,
        feedback_id: feedbackId,
        status,
      })
      // Reload queue data after review
      await loadQueueData(selectedSession)
    } catch (err) {
      setError(getErrorMessage(err, `Failed to ${status} correction.`))
    } finally {
      setReviewingId(null)
    }
  }

  // Filter queues
  const pendingItems = queue.filter((item) => item.status === 'pending')
  const historyItems = queue.filter((item) => item.status !== 'pending')
  const currentItems = activeTab === 'pending' ? pendingItems : historyItems

  return (
    <div className="p-8 max-w-5xl">
      
      {/* Header */}
      <div className="mb-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Inbox size={20} className="text-blue-400" />
            <h1 className="text-2xl font-bold text-white">Review Queue</h1>
          </div>
          <p className="text-gray-400 text-sm">
            Review manual adjustments, validate machine constraints, and download feedback logs.
          </p>
        </div>

        {/* Export Dropdown */}
        {selectedSession && (
          <div className="relative">
            <button
              onClick={() => setExportOpen(!exportOpen)}
              className="btn-primary flex items-center gap-2 text-sm px-4 py-2"
            >
              <Download size={14} />
              Export Annotations
              <ChevronDown size={14} />
            </button>
            {exportOpen && (
              <div className="absolute right-0 mt-2 w-48 bg-slate-900 border border-slate-700/80 rounded-lg shadow-xl overflow-hidden z-30">
                <a
                  href={getFeedbackExportUrl(selectedSession, 'csv')}
                  onClick={() => setExportOpen(false)}
                  className="flex items-center gap-2 px-4 py-3 text-sm text-slate-300 hover:bg-slate-800 hover:text-slate-100 transition-colors border-b border-slate-800"
                >
                  <FileSpreadsheet size={14} className="text-emerald-400" /> Export as CSV
                </a>
                <a
                  href={getFeedbackExportUrl(selectedSession, 'json')}
                  onClick={() => setExportOpen(false)}
                  className="flex items-center gap-2 px-4 py-3 text-sm text-slate-300 hover:bg-slate-800 hover:text-slate-100 transition-colors"
                >
                  <FileCode size={14} className="text-amber-400" /> Export as JSON
                </a>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Session selector */}
      <div className="card p-4 mb-6">
        <div>
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
              onChange={(e) => navigate(`/review-queue/${e.target.value}`)}
              className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 w-full md:w-96 focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              {sessions.map((s) => (
                <option key={s.id} value={s.id}>
                  #{s.id} — {s.filename} ({s.total_requirements} reqs, {s.total_clusters} clusters)
                </option>
              ))}
            </select>
          )}
        </div>
      </div>

      {/* Metrics Banner */}
      {selectedSession && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
          <div className="card p-3 bg-slate-800/40 border border-slate-700/50">
            <div className="text-xs text-gray-500">Pending Review</div>
            <div className="text-xl font-bold text-blue-400 mt-1">{pendingItems.length}</div>
          </div>
          <div className="card p-3 bg-slate-800/40 border border-slate-700/50">
            <div className="text-xs text-gray-500">Approved</div>
            <div className="text-xl font-bold text-emerald-400 mt-1">
              {queue.filter((i) => i.status === 'approved').length}
            </div>
          </div>
          <div className="card p-3 bg-slate-800/40 border border-slate-700/50">
            <div className="text-xs text-gray-500">Rejected</div>
            <div className="text-xl font-bold text-red-400 mt-1">
              {queue.filter((i) => i.status === 'rejected').length}
            </div>
          </div>
          <div className="card p-3 bg-slate-800/40 border border-slate-700/50">
            <div className="text-xs text-gray-500">ML Constraints</div>
            <div className="text-xl font-bold text-slate-300 mt-1">
              {constraintsInfo.constraint_pairs?.length || 0}
            </div>
          </div>
        </div>
      )}

      {/* Error alert */}
      {error && (
        <div className="card p-3 mb-4 border-amber-900/30 bg-amber-950/10 flex items-start gap-2">
          <AlertTriangle size={14} className="text-amber-400 mt-0.5 flex-shrink-0" />
          <p className="text-sm text-amber-300/80">{error}</p>
        </div>
      )}

      {/* Conflict Validation Section */}
      {selectedSession && constraintsInfo.has_conflicts && (
        <div className="card p-4 mb-6 border-red-900/40 bg-red-950/10">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle size={16} className="text-red-400" />
            <h2 className="text-sm font-semibold text-red-300">
              Constraint Conflict Detected ({constraintsInfo.conflicts.length})
            </h2>
          </div>
          <div className="space-y-2">
            {constraintsInfo.conflicts.map((conflict, index) => (
              <div
                key={index}
                className="p-3 text-xs bg-slate-950/50 rounded border border-red-900/20 text-red-200/80 space-y-1.5"
              >
                <div>{conflict.message}</div>
                <div className="flex flex-col gap-1 pl-2 border-l border-red-900/30 text-gray-400">
                  <div>
                    <span className="text-red-400 font-semibold">{conflict.requirement_a_label}:</span> "{conflict.requirement_a_text}"
                  </div>
                  <div>
                    <span className="text-red-400 font-semibold">{conflict.requirement_b_label}:</span> "{conflict.requirement_b_text}"
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 mb-4 bg-gray-900 rounded-lg p-1 w-fit">
        <button
          onClick={() => setActiveTab('pending')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm transition-colors ${
            activeTab === 'pending'
              ? 'bg-gray-800 text-white font-medium'
              : 'text-gray-400 hover:text-gray-200'
          }`}
        >
          Pending adjustments
          {pendingItems.length > 0 && (
            <span className="text-xs bg-blue-500 text-white px-1.5 py-0.5 rounded-full ml-1 font-semibold">
              {pendingItems.length}
            </span>
          )}
        </button>
        <button
          onClick={() => setActiveTab('history')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm transition-colors ${
            activeTab === 'history'
              ? 'bg-gray-800 text-white font-medium'
              : 'text-gray-400 hover:text-gray-200'
          }`}
        >
          Review history
        </button>
      </div>

      {/* Queue items */}
      <div className="card overflow-hidden">
        {currentItems.length === 0 ? (
          <div className="p-8 text-center text-gray-500 text-sm">
            {activeTab === 'pending' ? 'No pending cluster adjustments in queue.' : 'No reviewed adjustments found.'}
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-800">
              <thead className="bg-slate-800/40">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">Requirement</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">Source</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">Target</th>
                  <th className="px-4 py-3 text-center text-xs font-semibold uppercase tracking-wider text-slate-400">Conf.</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">Analyst / Notes</th>
                  {activeTab === 'pending' ? (
                    <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-slate-400">Actions</th>
                  ) : (
                    <th className="px-4 py-3 text-center text-xs font-semibold uppercase tracking-wider text-slate-400">Status</th>
                  )}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800/60 bg-slate-900/10">
                {currentItems.map((item) => {
                  const req = reqMap[item.requirement_id] || {}
                  const isExpanded = expandedId === item.id
                  return (
                    <React.Fragment key={item.id}>
                      <tr
                        onClick={() => setExpandedId(isExpanded ? null : item.id)}
                        className="hover:bg-slate-800/30 cursor-pointer transition-colors"
                      >
                        {/* Requirement ID */}
                        <td className="px-4 py-3 text-sm text-slate-200 font-semibold whitespace-nowrap">
                          {req.req_id || `REQ-${req.id || item.requirement_id}`}
                          <span className="text-[10px] text-gray-500 block">Click to view text</span>
                        </td>
                        {/* Source Cluster */}
                        <td className="px-4 py-3 text-sm text-gray-400 truncate max-w-[120px]">
                          {item.previous_cluster_id !== null && item.previous_cluster_id !== -1
                            ? clusterInfo[item.previous_cluster_id] || `Cluster ${item.previous_cluster_id}`
                            : 'Noise / Unclustered'}
                        </td>
                        {/* Target Cluster */}
                        <td className="px-4 py-3 text-sm text-blue-400 truncate max-w-[120px] font-medium">
                          {item.new_cluster_id !== null && item.new_cluster_id !== -1
                            ? clusterInfo[item.new_cluster_id] || `Cluster ${item.new_cluster_id}`
                            : 'Noise / Unclustered'}
                        </td>
                        {/* Confidence */}
                        <td className="px-4 py-3 text-center text-sm font-semibold text-slate-300">
                          {Math.round(item.confidence_score * 100)}%
                        </td>
                        {/* Comments & Analyst */}
                        <td className="px-4 py-3 text-sm text-gray-400 max-w-[200px] truncate">
                          <span className="font-semibold text-xs text-gray-500 block truncate">{item.applied_by}</span>
                          <span className="text-xs">{item.comments || 'No comment provided.'}</span>
                        </td>
                        {/* Action buttons or status indicator */}
                        {activeTab === 'pending' ? (
                          <td className="px-4 py-3 text-right whitespace-nowrap" onClick={(e) => e.stopPropagation()}>
                            <div className="flex justify-end gap-2">
                              <button
                                onClick={() => handleReview(item.id, 'approved')}
                                disabled={reviewingId !== null}
                                className="p-1 px-2.5 rounded bg-emerald-950/40 hover:bg-emerald-600/80 text-emerald-400 hover:text-white border border-emerald-900/30 text-xs transition-colors flex items-center gap-1"
                              >
                                Approve
                              </button>
                              <button
                                onClick={() => handleReview(item.id, 'rejected')}
                                disabled={reviewingId !== null}
                                className="p-1 px-2.5 rounded bg-red-950/40 hover:bg-red-600/80 text-red-400 hover:text-white border border-red-900/30 text-xs transition-colors flex items-center gap-1"
                              >
                                Reject
                              </button>
                            </div>
                          </td>
                        ) : (
                          <td className="px-4 py-3 text-center whitespace-nowrap">
                            <span
                              className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold ${
                                item.status === 'approved'
                                  ? 'bg-emerald-950 text-emerald-400 border border-emerald-900/40'
                                  : 'bg-red-950 text-red-400 border border-red-900/40'
                              }`}
                            >
                              {item.status === 'approved' ? (
                                <>
                                  <CheckCircle size={10} /> Approved
                                </>
                              ) : (
                                <>
                                  <XCircle size={10} /> Rejected
                                </>
                              )}
                            </span>
                          </td>
                        )}
                      </tr>
                      {/* Expanded requirement text view */}
                      {isExpanded && (
                        <tr>
                          <td colSpan={6} className="px-6 py-4 bg-slate-950/40">
                            <div className="space-y-2 text-sm">
                              <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                                Requirement Text
                              </div>
                              <p className="text-slate-200 leading-relaxed italic">
                                "{req.text || 'N/A'}"
                              </p>
                              {req.module && (
                                <div className="text-xs text-gray-500">
                                  Module: {req.module} {req.section ? `| Section: ${req.section}` : ''}
                                </div>
                              )}
                              {item.comments && (
                                <div className="mt-3 p-3 bg-slate-950 rounded border border-slate-800 text-xs text-gray-400 space-y-1">
                                  <div className="font-semibold text-slate-300">Correction annotation:</div>
                                  <p>"{item.comments}"</p>
                                </div>
                              )}
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

    </div>
  )
}
