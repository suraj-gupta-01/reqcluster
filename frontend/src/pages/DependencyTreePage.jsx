import { useState, useEffect, useRef, useCallback } from 'react'
import { Workflow, Loader, RefreshCw, AlertTriangle, GitBranch, ArrowRight } from 'lucide-react'
import {
  getSessions, generateDependencies, getDependencies, getErrorMessage,
} from '../utils/api.js'
import { getClusterColor } from '../utils/colors.js'

const RELATION_COLORS = {
  data: '#2fbcaa',
  sequential: '#f59e0b',
  hierarchical: '#0ea5e9',
  reference: '#94a3b8',
}
const RELATION_LABEL = {
  data: 'Data (output → input)',
  sequential: 'Sequential (precondition)',
  hierarchical: 'Hierarchical (parent → child)',
  reference: 'Explicit reference',
}

export default function DependencyTreePage() {
  const plotRef = useRef(null)
  const [sessions, setSessions] = useState([])
  const [selected, setSelected] = useState(null)
  const [data, setData] = useState(null)
  const [loadingSessions, setLoadingSessions] = useState(true)
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState(null)
  const [selectedNode, setSelectedNode] = useState(null)

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
      const res = await getDependencies(sid)
      setData(res)
      setError(null)
    } catch {
      setData(null) // not generated yet
    }
  }, [])

  useEffect(() => { if (selected) load(selected) }, [selected, load])

  const handleGenerate = async () => {
    if (!selected) return
    setGenerating(true); setError(null)
    try {
      const res = await generateDependencies({ session_id: selected })
      setData(res)
    } catch (err) {
      setError(getErrorMessage(err, 'Failed to generate dependency tree.'))
    } finally { setGenerating(false) }
  }

  // Render the tree with Plotly: x = dependency level, y = spread within level.
  useEffect(() => {
    if (!data || !plotRef.current) return
    const el = plotRef.current
    let disposed = false
    const draw = async () => {
      const Plotly = (await import('plotly.js-dist-min')).default
      if (disposed || !el) return
      const { nodes, edges } = data
      const byLevel = {}
      nodes.forEach(n => { (byLevel[n.level] ||= []).push(n) })
      const pos = {}
      Object.entries(byLevel).forEach(([lvl, group]) => {
        const L = group.length
        group.forEach((n, i) => { pos[n.id] = { x: Number(lvl), y: i - (L - 1) / 2 } })
      })

      const edgeTraces = Object.keys(RELATION_COLORS).map(rel => {
        const ex = [], ey = []
        edges.filter(e => e.relation === rel).forEach(e => {
          const s = pos[e.source], t = pos[e.target]
          if (!s || !t) return
          ex.push(s.x, t.x, null); ey.push(s.y, t.y, null)
        })
        return {
          type: 'scatter', mode: 'lines', x: ex, y: ey, name: RELATION_LABEL[rel],
          line: { width: 1, color: RELATION_COLORS[rel], shape: 'spline' },
          hoverinfo: 'none', opacity: 0.7,
        }
      })

      const nodeTrace = {
        type: 'scatter', mode: 'markers', name: 'Requirements', showlegend: false,
        x: nodes.map(n => pos[n.id].x), y: nodes.map(n => pos[n.id].y),
        text: nodes.map(n => n.node_id),
        customdata: nodes.map(n => ({ id: n.id, req_id: n.node_id, text: n.requirement_text, cluster_id: n.cluster_id, level: n.level })),
        hovertemplate: '<b>%{customdata.req_id}</b><br>%{customdata.text}<br><i>Level %{customdata.level}</i><extra></extra>',
        marker: {
          size: 12, color: nodes.map(n => getClusterColor(n.cluster_id)),
          line: { width: 1.5, color: 'rgba(0,0,0,0.45)' },
        },
      }

      const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#9ca3af', size: 11, family: 'Plus Jakarta Sans, sans-serif' },
        margin: { l: 20, r: 20, t: 20, b: 20 },
        xaxis: { title: 'Dependency level', gridcolor: 'rgba(255,255,255,0.05)', zeroline: false },
        yaxis: { showgrid: false, zeroline: false, showticklabels: false },
        legend: { bgcolor: 'rgba(17,22,22,0.7)', bordercolor: 'rgba(255,255,255,0.08)', borderwidth: 1, orientation: 'h', y: 1.06, font: { size: 10 } },
        hoverlabel: { bgcolor: '#11161a', bordercolor: '#1f2937', font: { color: '#e5e7eb', size: 12 }, align: 'left' },
        dragmode: 'pan',
      }
      el.removeAllListeners?.('plotly_click')
      Plotly.react(el, [...edgeTraces, nodeTrace], layout, { responsive: true, displaylogo: false, scrollZoom: true, modeBarButtonsToRemove: ['select2d', 'lasso2d'] })
      el.on('plotly_click', (ev) => { if (ev.points?.[0]?.customdata) setSelectedNode(ev.points[0].customdata) })
    }
    draw()
    return () => { disposed = true }
  }, [data])

  useEffect(() => {
    const el = plotRef.current
    return () => { if (el) import('plotly.js-dist-min').then(({ default: P }) => P.purge(el)).catch(() => {}) }
  }, [])

  const stats = data?.stats || {}
  const grouping = data?.rationale?.grouping || []

  return (
    <div className="p-8 max-w-6xl animate-fade-up">
      <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Workflow size={20} className="text-brand-400" />
            <h1 className="text-2xl font-bold text-white tracking-tight">Dependency Tree</h1>
          </div>
          <p className="text-gray-400 text-sm max-w-xl">
            Hierarchical and sequential relationships inferred from each requirement's inputs,
            preconditions, and outputs, with a generated rationale document.
          </p>
        </div>
        <button onClick={handleGenerate} disabled={!selected || generating} className="btn-primary text-sm">
          {generating ? <><Loader size={14} className="animate-spin" /> Analyzing…</> : <><RefreshCw size={14} /> Build dependency tree</>}
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

      {error && (
        <div className="card p-3 mb-4 border-amber-900/30 flex items-start gap-2">
          <AlertTriangle size={14} className="text-amber-400 mt-0.5 flex-shrink-0" />
          <p className="text-sm text-amber-300/80">{error}</p>
        </div>
      )}

      {!data ? (
        <div className="card p-12 text-center text-gray-500 text-sm">
          <GitBranch size={28} className="mx-auto mb-3 text-gray-600" />
          No dependency tree yet. Click <span className="text-brand-400 font-medium">Build dependency tree</span> to analyze this session.
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
            {[
              ['Requirements', stats.n_nodes],
              ['Dependencies', stats.n_edges],
              ['Max depth', stats.max_depth],
              ['Roots', stats.root_count],
            ].map(([label, val]) => (
              <div key={label} className="stat-card">
                <span className="text-xs text-gray-500 uppercase tracking-wider">{label}</span>
                <span className="text-2xl font-bold text-white font-mono">{val ?? '—'}</span>
              </div>
            ))}
          </div>

          <div className="grid lg:grid-cols-3 gap-4">
            <div className="lg:col-span-2 card overflow-hidden" style={{ height: 460 }}>
              <div ref={plotRef} style={{ width: '100%', height: '100%' }} />
            </div>

            <div className="card p-4 overflow-y-auto" style={{ height: 460 }}>
              {selectedNode ? (
                <div className="space-y-2">
                  <span className="text-xs font-mono text-brand-400">{selectedNode.req_id}</span>
                  <p className="text-sm text-gray-200 leading-relaxed">{selectedNode.text}</p>
                  <div className="text-xs text-gray-500">Level {selectedNode.level} · Cluster {selectedNode.cluster_id === -1 ? 'Noise' : selectedNode.cluster_id}</div>
                </div>
              ) : (
                <>
                  <h2 className="text-sm font-semibold text-gray-300 mb-3">Grouping rationale</h2>
                  <div className="space-y-3">
                    {grouping.map(g => (
                      <div key={g.cluster_id} className="border-l-2 pl-3" style={{ borderColor: getClusterColor(g.cluster_id) }}>
                        <div className="text-sm font-medium text-gray-200">{g.label}</div>
                        <p className="text-xs text-gray-400 mt-0.5 leading-relaxed">{g.rationale}</p>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          </div>

          {data?.rationale?.dependencies?.length > 0 && (
            <div className="card mt-4 overflow-hidden">
              <div className="px-4 py-3 border-b border-white/[0.06] text-sm font-semibold text-gray-300">
                Dependency justifications
              </div>
              <div className="divide-y divide-white/[0.04] max-h-72 overflow-y-auto">
                {data.rationale.dependencies.slice(0, 100).map((d, i) => (
                  <div key={i} className="px-4 py-2.5 flex items-center gap-3 text-sm">
                    <span className="font-mono text-xs text-gray-400 flex items-center gap-1.5 flex-shrink-0">
                      {d.source_req_id}<ArrowRight size={12} className="text-gray-600" />{d.target_req_id}
                    </span>
                    <span className="badge text-[10px]" style={{ backgroundColor: (RELATION_COLORS[d.relation] || '#94a3b8') + '22', color: RELATION_COLORS[d.relation] || '#94a3b8' }}>
                      {d.relation}
                    </span>
                    <span className="text-xs text-gray-500 truncate">{d.justification}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
