import { useState, useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import { Brain, Sparkles, AlertTriangle, TrendingUp, Gauge, PencilLine } from 'lucide-react'
import {
  getUncertainty, startRecluster, getQuality, getClusters, submitFeedback,
} from '../api/endpoints.js'
import { useApi } from '../hooks/useApi.js'
import { useJob } from '../hooks/useJob.js'
import { useSession } from '../state/SessionContext.jsx'
import { getErrorMessage, isConflict } from '../utils/format.js'
import { getClusterColor } from '../utils/colors.js'
import MoveToClusterModal from '../components/MoveToClusterModal.jsx'
import JobProgress from '../components/JobProgress.jsx'

const EMPTY = []
const CONFLICT_MSG = 'Data changed (another edit or a running clustering job). Refresh and try again.'

export default function ActiveLearningPage() {
  const chartRef = useRef(null)
  const { sessionId: selected, reloadSession } = useSession()
  const [modalReq, setModalReq] = useState(null)
  const [jobId, setJobId] = useState(null)
  const [actionError, setActionError] = useState(null)
  const [conflict, setConflict] = useState(false)
  const [notice, setNotice] = useState(null)

  const { data, error: loadError, reload } = useApi(async (o) => {
    if (!selected) return null
    const [queue, history, clusters] = await Promise.all([
      getUncertainty(selected, 25, o), getQuality(selected, o), getClusters(selected, o),
    ])
    return { queue, history, clusters }
  }, [selected])
  const queue = data?.queue || []
  const history = data?.history ?? EMPTY
  const clusters = data?.clusters || []
  const error = actionError || (loadError && getErrorMessage(loadError, 'Failed to load active-learning data.'))
  const running = jobId != null

  const refresh = () => { setConflict(false); setActionError(null); reload(); reloadSession() }

  const { job, cancel } = useJob(jobId, (j) => {
    setJobId(null)
    if (j.status === 'succeeded') setNotice(`Constrained re-clustering finished${j.result?.moved != null ? `; ${j.result.moved} requirements moved` : ''}.`)
    else if (j.status === 'failed') setActionError(j.error || 'Constrained re-clustering failed.')
    reload(); reloadSession()
  })

  const handleReassign = async (payload) => {
    if (!selected || !modalReq) return
    try {
      await submitFeedback(selected, { requirement_id: modalReq.id, ...payload })
      setModalReq(null)
      setNotice('Correction applied. Use Apply constraints to re-cluster with all current constraints.')
      reload()
    } catch (err) {
      setModalReq(null)
      if (isConflict(err)) setConflict(true)
      else setActionError(getErrorMessage(err, 'Failed to submit correction.'))
    }
  }

  const handleRecluster = async () => {
    if (!selected) return
    setActionError(null); setNotice(null); setConflict(false)
    try {
      const j = await startRecluster(selected)
      setJobId(j.id)
    } catch (err) {
      if (isConflict(err)) setConflict(true)
      else setActionError(getErrorMessage(err, 'Constrained re-clustering failed.'))
    }
  }

  // Quality history line chart.
  useEffect(() => {
    if (!chartRef.current || history.length === 0) return
    const el = chartRef.current
    let disposed = false
    const draw = async () => {
      const Plotly = (await import('plotly.js-dist-min')).default
      if (disposed || !el) return
      const x = history.map(h => `#${h.revision}`)
      const traces = [
        { x, y: history.map(h => h.silhouette), name: 'Silhouette', type: 'scatter', mode: 'lines+markers', line: { color: '#2fbcaa', width: 2 }, yaxis: 'y' },
        { x, y: history.map(h => h.noise_count), name: 'Noise count', type: 'scatter', mode: 'lines+markers', line: { color: '#f59e0b', width: 2 }, yaxis: 'y2' },
      ]
      const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#9ca3af', size: 11, family: 'Plus Jakarta Sans, sans-serif' },
        margin: { l: 44, r: 44, t: 16, b: 36 },
        xaxis: { gridcolor: 'rgba(255,255,255,0.05)' },
        yaxis: { title: 'Silhouette', gridcolor: 'rgba(255,255,255,0.05)', titlefont: { color: '#2fbcaa' } },
        yaxis2: { title: 'Noise count', overlaying: 'y', side: 'right', titlefont: { color: '#f59e0b' }, showgrid: false },
        legend: { orientation: 'h', y: 1.15, font: { size: 10 } },
      }
      Plotly.react(el, traces, layout, { responsive: true, displaylogo: false, displayModeBar: false })
    }
    draw()
    return () => { disposed = true }
  }, [history])

  useEffect(() => {
    const el = chartRef.current
    return () => { if (el) import('plotly.js-dist-min').then(({ default: P }) => P.purge(el)).catch(() => {}) }
  }, [])

  return (
    <div className="p-8 max-w-6xl animate-fade-up">
      <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Brain size={20} className="text-brand-400" />
            <h1 className="text-2xl font-bold text-white tracking-tight">Active Learning</h1>
          </div>
          <p className="text-gray-400 text-sm max-w-xl">
            Review the least-certain assignments, then fold your accepted corrections back into the
            clustering as must-link / cannot-link constraints. A submitted move applies immediately;
            "Apply constraints" re-clusters using all current constraints (pending and approved).
          </p>
        </div>
        <button onClick={handleRecluster} disabled={!selected || running} className="btn-primary text-sm">
          <Sparkles size={14} /> Apply constraints
        </button>
      </div>

      {!selected && (
        <div className="card p-4 mb-6 text-sm">
          <Link to="/" className="text-brand-400 hover:underline">Select or upload a session first</Link>
        </div>
      )}
      {running && <div className="mb-4"><JobProgress job={job} onCancel={cancel} title="Re-clustering" /></div>}
      {conflict && (
        <div className="card p-3 mb-4 border-amber-900/30 flex items-center gap-3">
          <AlertTriangle size={14} className="text-amber-400 flex-shrink-0" />
          <p className="text-sm text-amber-300/80 flex-1">{CONFLICT_MSG}</p>
          <button onClick={refresh} className="btn-secondary text-xs py-1 px-3">Refresh</button>
        </div>
      )}

      {notice && <div className="card p-3 mb-4 border-brand-900/30"><p className="text-sm text-brand-300/90">{notice}</p></div>}
      {error && (
        <div className="card p-3 mb-4 border-amber-900/30 flex items-start gap-2">
          <AlertTriangle size={14} className="text-amber-400 mt-0.5 flex-shrink-0" />
          <p className="text-sm text-amber-300/80">{error}</p>
        </div>
      )}

      <div className="grid lg:grid-cols-5 gap-4">
        <div className="lg:col-span-3 card overflow-hidden">
          <div className="px-4 py-3 border-b border-white/[0.06] flex items-center gap-2">
            <Gauge size={14} className="text-gray-500" />
            <h2 className="text-sm font-semibold text-gray-300">Most uncertain requirements</h2>
          </div>
          <div className="divide-y divide-white/[0.04] max-h-[26rem] overflow-y-auto">
            {queue.length === 0 ? (
              <div className="p-8 text-center text-gray-500 text-sm">No data. Select a clustered session.</div>
            ) : queue.map(item => (
              <div key={item.requirement_id} className="px-4 py-3 flex items-center gap-3 group">
                <span className="font-mono text-xs text-gray-400 w-16 flex-shrink-0">{item.req_id}</span>
                <p className="text-sm text-gray-300 flex-1 truncate">{item.text}</p>
                <div className="flex items-center gap-1.5 flex-shrink-0">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: getClusterColor(item.cluster_id) }} />
                  <div className="w-16 h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
                    <div className="h-full rounded-full" style={{ width: `${Math.round(item.uncertainty * 100)}%`, background: '#f59e0b' }} />
                  </div>
                  <span className="text-xs font-mono text-gray-500 w-9 text-right">{Math.round(item.uncertainty * 100)}%</span>
                </div>
                <button
                  onClick={() => setModalReq({ id: item.requirement_id, req_id: item.req_id, text: item.text, cluster_id: item.cluster_id })}
                  className="flex-shrink-0 p-1.5 rounded-lg text-gray-500 hover:text-brand-300 hover:bg-white/[0.05] transition-colors opacity-0 group-hover:opacity-100"
                  title="Reassign cluster"
                >
                  <PencilLine size={14} />
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className="lg:col-span-2 card p-4">
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp size={14} className="text-gray-500" />
            <h2 className="text-sm font-semibold text-gray-300">Quality across iterations</h2>
          </div>
          {history.length === 0 ? (
            <div className="text-center text-gray-500 text-sm py-12">
              No iterations yet. Submit feedback corrections, then apply constraints.
            </div>
          ) : (
            <div ref={chartRef} style={{ width: '100%', height: 300 }} />
          )}
        </div>
      </div>

      <MoveToClusterModal
        isOpen={!!modalReq}
        onClose={() => setModalReq(null)}
        requirement={modalReq}
        clusters={clusters}
        onSubmit={handleReassign}
      />
    </div>
  )
}
