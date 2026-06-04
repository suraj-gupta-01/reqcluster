import { useState, useEffect, useRef, useCallback } from 'react'
import { Brain, Loader, Sparkles, AlertTriangle, TrendingUp, Gauge } from 'lucide-react'
import {
  getSessions, getUncertaintyQueue, runConstrainedClustering, getQualityHistory, getErrorMessage,
} from '../utils/api.js'
import { getClusterColor } from '../utils/colors.js'

export default function ActiveLearningPage() {
  const chartRef = useRef(null)
  const [sessions, setSessions] = useState([])
  const [selected, setSelected] = useState(null)
  const [queue, setQueue] = useState([])
  const [history, setHistory] = useState([])
  const [loadingSessions, setLoadingSessions] = useState(true)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState(null)
  const [notice, setNotice] = useState(null)

  useEffect(() => {
    getSessions()
      .then(list => {
        const done = list.filter(s => s.status === 'done')
        setSessions(done)
        if (done.length && !selected) setSelected(done[0].id)
      })
      .catch(() => setError('Failed to load sessions.'))
      .finally(() => setLoadingSessions(false))
  }, [])

  const load = useCallback(async (sid) => {
    try {
      const [q, h] = await Promise.all([getUncertaintyQueue(sid, 25), getQualityHistory(sid)])
      setQueue(q.queue || [])
      setHistory(h.history || [])
      setError(null)
    } catch (err) {
      setError(getErrorMessage(err, 'Failed to load active-learning data.'))
    }
  }, [])

  useEffect(() => { if (selected) load(selected) }, [selected, load])

  const handleRecluster = async () => {
    if (!selected) return
    setRunning(true); setError(null); setNotice(null)
    try {
      const res = await runConstrainedClustering(selected)
      const moved = res.constraints.points_moved_must_link + res.constraints.points_moved_cannot_link
      setNotice(`Iteration ${res.iteration}: applied ${res.constraints.must_link_pairs} must-link and ${res.constraints.cannot_link_pairs} cannot-link constraints, moved ${moved} requirement(s).`)
      await load(selected)
    } catch (err) {
      setError(getErrorMessage(err, 'Constrained re-clustering failed.'))
    } finally { setRunning(false) }
  }

  // Quality history line chart.
  useEffect(() => {
    if (!chartRef.current || history.length === 0) return
    const el = chartRef.current
    let disposed = false
    const draw = async () => {
      const Plotly = (await import('plotly.js-dist-min')).default
      if (disposed || !el) return
      const x = history.map(h => `#${h.iteration}`)
      const traces = [
        { x, y: history.map(h => h.silhouette), name: 'Silhouette', type: 'scatter', mode: 'lines+markers', line: { color: '#2fbcaa', width: 2 }, yaxis: 'y' },
        { x, y: history.map(h => h.noise_rate), name: 'Noise rate', type: 'scatter', mode: 'lines+markers', line: { color: '#f59e0b', width: 2 }, yaxis: 'y2' },
      ]
      const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#9ca3af', size: 11, family: 'Plus Jakarta Sans, sans-serif' },
        margin: { l: 44, r: 44, t: 16, b: 36 },
        xaxis: { gridcolor: 'rgba(255,255,255,0.05)' },
        yaxis: { title: 'Silhouette', gridcolor: 'rgba(255,255,255,0.05)', titlefont: { color: '#2fbcaa' } },
        yaxis2: { title: 'Noise rate', overlaying: 'y', side: 'right', titlefont: { color: '#f59e0b' }, showgrid: false },
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
    <div className="p-8 max-w-5xl animate-fade-up">
      <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Brain size={20} className="text-brand-400" />
            <h1 className="text-2xl font-bold text-white tracking-tight">Active Learning</h1>
          </div>
          <p className="text-gray-400 text-sm max-w-xl">
            Review the least-certain assignments, then fold your accepted corrections back into the
            clustering as must-link / cannot-link constraints.
          </p>
        </div>
        <button onClick={handleRecluster} disabled={!selected || running} className="btn-primary text-sm">
          {running ? <><Loader size={14} className="animate-spin" /> Re-clustering…</> : <><Sparkles size={14} /> Apply constraints</>}
        </button>
      </div>

      <div className="card p-4 mb-6">
        <label className="text-xs text-gray-500 block mb-1.5">Session</label>
        {loadingSessions ? (
          <div className="flex items-center gap-2 text-gray-400 text-sm"><Loader size={14} className="animate-spin" /> Loading…</div>
        ) : sessions.length === 0 ? (
          <div className="text-sm text-gray-500">No completed sessions. Run clustering first.</div>
        ) : (
          <select value={selected || ''} onChange={e => setSelected(parseInt(e.target.value))} className="input text-sm w-full md:w-96">
            {sessions.map(s => <option key={s.id} value={s.id}>#{s.id} — {s.filename} ({s.total_requirements} reqs)</option>)}
          </select>
        )}
      </div>

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
              <div key={item.index} className="px-4 py-3 flex items-center gap-3">
                <span className="font-mono text-xs text-gray-400 w-16 flex-shrink-0">{item.req_id}</span>
                <p className="text-sm text-gray-300 flex-1 truncate">{item.text}</p>
                <div className="flex items-center gap-1.5 flex-shrink-0">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: getClusterColor(item.cluster_id) }} />
                  <div className="w-16 h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
                    <div className="h-full rounded-full" style={{ width: `${Math.round(item.uncertainty * 100)}%`, background: '#f59e0b' }} />
                  </div>
                  <span className="text-xs font-mono text-gray-500 w-9 text-right">{Math.round(item.uncertainty * 100)}%</span>
                </div>
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
    </div>
  )
}
