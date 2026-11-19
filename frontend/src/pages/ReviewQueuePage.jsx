import React, { useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Inbox,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Download,
  ChevronDown,
  FileSpreadsheet,
  FileCode,
} from 'lucide-react'
import {
  getFeedback,
  reviewFeedback,
  getConstraints,
  feedbackExportUrl,
  getClusters,
} from '../api/endpoints.js'
import { useApi } from '../hooks/useApi.js'
import { useSession } from '../state/SessionContext.jsx'
import { getErrorMessage, isConflict } from '../utils/format.js'

const CONFLICT_MSG = 'Data changed (another edit or a running clustering job). Refresh and try again.'

export default function ReviewQueuePage() {
  const { sessionId } = useSession()
  const selectedSession = sessionId

  const [expandedId, setExpandedId] = useState(null)
  const [activeTab, setActiveTab] = useState('pending') // pending, history
  const [reviewingId, setReviewingId] = useState(null)
  const [actionError, setActionError] = useState(null)
  const [conflict, setConflict] = useState(false)
  const [exportOpen, setExportOpen] = useState(false)

  const { data, error: loadError, reload } = useApi(async (o) => {
    if (!sessionId) return null
    const [queue, constraints, clusters] = await Promise.all([
      getFeedback(sessionId, undefined, o),
      getConstraints(sessionId, o),
      getClusters(sessionId, o),
    ])
    return { queue, constraints, clusters }
  }, [sessionId])

  const queue = data?.queue || []
  const constraintsInfo = data?.constraints || { constraint_pairs: [], conflicts: [] }
  const clusterInfo = Object.fromEntries((data?.clusters || []).map((c) => [c.cluster_id, c.label]))
  const error = actionError || (loadError && getErrorMessage(loadError, 'Failed to load queue details.'))

  const refresh = () => { setConflict(false); setActionError(null); reload() }

  const handleReview = async (feedbackId, status) => {
    if (!selectedSession) return
    setReviewingId(feedbackId)
    setActionError(null); setConflict(false)
    try {
      await reviewFeedback(selectedSession, feedbackId, status === 'approved' ? 'approve' : 'reject')
      reload()
    } catch (err) {
      if (isConflict(err)) setConflict(true)
      else setActionError(getErrorMessage(err, `Failed to ${status} correction.`))
    } finally {
      setReviewingId(null)
    }
  }

  // Filter queues
  const pendingItems = queue.filter((item) => item.status === 'pending')
  const historyItems = queue.filter((item) => item.status !== 'pending')
  const currentItems = activeTab === 'pending' ? pendingItems : historyItems

  return (
    <div className="p-8 max-w-6xl">
      
      {/* Header */}
      <div className="mb-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Inbox size={20} className="text-brand-400" />
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
                  href={feedbackExportUrl(selectedSession, 'csv')}
                  onClick={() => setExportOpen(false)}
                  className="flex items-center gap-2 px-4 py-3 text-sm text-slate-300 hover:bg-slate-800 hover:text-slate-100 transition-colors border-b border-slate-800"
                >
                  <FileSpreadsheet size={14} className="text-emerald-400" /> Export as CSV
                </a>
                <a
                  href={feedbackExportUrl(selectedSession, 'json')}
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

      {!selectedSession && (
        <div className="card p-4 mb-6 text-sm">
          <Link to="/" className="text-brand-400 hover:underline">Select or upload a session first</Link>
        </div>
      )}

      {/* Metrics Banner */}
      {selectedSession && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
          <div className="card p-3 bg-slate-800/40 border border-slate-700/50">
            <div className="text-xs text-gray-500">Pending Review</div>
            <div className="text-xl font-bold text-brand-400 mt-1">{pendingItems.length}</div>
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

      {conflict && (
        <div className="card p-3 mb-4 border-amber-900/30 bg-amber-950/10 flex items-center gap-3">
          <AlertTriangle size={14} className="text-amber-400 flex-shrink-0" />
          <p className="text-sm text-amber-300/80 flex-1">{CONFLICT_MSG}</p>
          <button onClick={refresh} className="btn-secondary text-xs py-1 px-3">Refresh</button>
        </div>
      )}

      {/* Conflict Validation Section */}
      {selectedSession && constraintsInfo.conflicts.length > 0 && (
        <div className="card p-4 mb-6 border-red-900/40 bg-red-950/10">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle size={16} className="text-red-400" />
            <h2 className="text-sm font-semibold text-red-300">
              Constraint Conflict Detected ({constraintsInfo.conflicts.length})
            </h2>
          </div>
          <div className="space-y-2">
            {constraintsInfo.conflicts.map((cf, index) => (
              <div
                key={index}
                className="p-3 text-xs bg-slate-950/50 rounded border border-red-900/20 text-red-200/80 space-y-1.5"
              >
                <div>Requirements {cf.a_req_id ?? cf.a} and {cf.b_req_id ?? cf.b} have both {cf.kinds.join(' and ')} constraints.</div>
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
            <span className="text-xs bg-brand-500 text-white px-1.5 py-0.5 rounded-full ml-1 font-semibold">
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
                  const req = item
                  const isExpanded = expandedId === item.id
                  return (
                    <React.Fragment key={item.id}>
                      <tr
                        onClick={() => setExpandedId(isExpanded ? null : item.id)}
                        className="hover:bg-slate-800/30 cursor-pointer transition-colors"
                      >
                        {/* Requirement ID */}
                        <td className="px-4 py-3 text-sm text-slate-200 font-semibold whitespace-nowrap">
                          {item.req_id}
                          <span className="text-[10px] text-gray-500 block">Click to view text</span>
                        </td>
                        {/* Source Cluster */}
                        <td className="px-4 py-3 text-sm text-gray-400 truncate max-w-[120px]">
                          {item.old_cluster_id !== null && item.old_cluster_id !== -1
                            ? clusterInfo[item.old_cluster_id] || `Cluster ${item.old_cluster_id}`
                            : 'Noise / Unclustered'}
                        </td>
                        {/* Target Cluster */}
                        <td className="px-4 py-3 text-sm text-brand-400 truncate max-w-[120px] font-medium">
                          {item.new_cluster_id !== null && item.new_cluster_id !== -1
                            ? clusterInfo[item.new_cluster_id] || `Cluster ${item.new_cluster_id}`
                            : 'Noise / Unclustered'}
                        </td>
                        {/* Confidence */}
                        <td className="px-4 py-3 text-center text-sm font-semibold text-slate-300">
                          {Math.round(item.confidence * 100)}%
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
