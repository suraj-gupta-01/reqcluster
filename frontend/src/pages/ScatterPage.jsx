import { useState, useEffect, useRef, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { Loader, Info } from 'lucide-react'
import { getRequirements, getClusters, getFeedback } from '../api/endpoints.js'
import { useApi } from '../hooks/useApi.js'
import { useSession } from '../state/SessionContext.jsx'
import { getErrorMessage } from '../utils/format.js'
import SessionGate from '../components/SessionGate.jsx'
import { getClusterColor } from '../utils/colors.js'

export default function ScatterPage() {
  return <SessionGate><Scatter /></SessionGate>
}

function Scatter() {
  const { sessionId, session } = useSession()
  const navigate = useNavigate()
  const plotRef = useRef(null)

  const reqQ = useApi(async (o) => {
    const items = []
    for (let page = 1; ; page++) {
      const r = await getRequirements(sessionId, { page, page_size: 1000 }, o)
      items.push(...r.items)
      if (items.length >= r.total || r.items.length === 0) return { items }
    }
  }, [sessionId])
  const clustersQ = useApi((o) => getClusters(sessionId, o), [sessionId])
  const fbQ = useApi((o) => getFeedback(sessionId, undefined, o), [sessionId])
  const requirements = useMemo(() => (reqQ.data?.items || []).filter(r => r.x != null), [reqQ.data])
  const clusters = useMemo(() => clustersQ.data || [], [clustersQ.data])
  const corrections = useMemo(() => fbQ.data || [], [fbQ.data])
  const loading = reqQ.loading || clustersQ.loading
  const error = reqQ.error || clustersQ.error
  const [selected, setSelected] = useState(null)
  const [filterCluster, setFilterCluster] = useState('all')
  const [showNoise, setShowNoise] = useState(true)

  const correctionMap = useMemo(() => {
    const map = {}
    corrections.forEach(c => {
      map[c.requirement_id] = c
    })
    return map
  }, [corrections])

  // Build Plotly traces once data is ready
  useEffect(() => {
    if (loading || !plotRef.current || requirements.length === 0) return

    const el = plotRef.current
    let disposed = false

    const loadPlotly = async () => {
      const Plotly = (await import('plotly.js-dist-min')).default
      if (disposed || !el) return

      // Group by cluster
      const clusterMap = {}
      requirements.forEach(r => {
        const cid = r.is_noise ? -1 : (r.cluster_id ?? -1)
        if (!clusterMap[cid]) clusterMap[cid] = []
        clusterMap[cid].push(r)
      })

      const clusterLabelMap = {}
      clusters.forEach(c => { clusterLabelMap[c.cluster_id] = c.label })

      const traces = []
      const sortedCids = Object.keys(clusterMap).map(Number).sort((a, b) => a - b)

      sortedCids.forEach(cid => {
        if (cid === -1 && !showNoise) return
        if (filterCluster !== 'all' && parseInt(filterCluster) !== cid) return

        const reqs = clusterMap[cid]
        const color = getClusterColor(cid)
        const name = cid === -1 ? 'Noise' : (clusterLabelMap[cid] || `Cluster ${cid}`)

        const sizes = reqs.map(r => {
          const hasCorrection = !!correctionMap[r.id]
          if (hasCorrection) return cid === -1 ? 10 : 12
          return cid === -1 ? 5 : 8
        })
        const linewidths = reqs.map(r => {
          const hasCorrection = !!correctionMap[r.id]
          return hasCorrection ? 2.5 : 1
        })
        const linecolors = reqs.map(r => {
          const corr = correctionMap[r.id]
          if (corr) {
            if (corr.status === 'pending') return '#3b82f6'
            if (corr.status === 'approved') return '#10b981'
            if (corr.status === 'rejected') return '#ef4444'
          }
          return 'rgba(0,0,0,0.3)'
        })

        traces.push({
          type: 'scattergl',
          mode: 'markers',
          name,
          x: reqs.map(r => r.x),
          y: reqs.map(r => r.y),
          text: reqs.map(r => r.req_id),
          customdata: reqs.map(r => ({
            id: r.id,
            req_id: r.req_id,
            text: r.text,
            cluster_id: r.cluster_id,
            prob: r.probability,
            correction: correctionMap[r.id] || null,
            attrs: Object.entries(r.attributes || {}).map(([k, v]) => `${k}: ${v}`).join(' | '),
          })),
          hovertemplate:
            '<b>%{customdata.req_id}</b><br>' +
            '%{customdata.text}<br>' +
            '<i>%{customdata.attrs}</i><br>' +
            '<i>Cluster: %{customdata.cluster_id}</i><br>' +
            '<i>Membership: %{customdata.prob:.2f}</i>' +
            '<extra></extra>',
          marker: {
            color,
            size: sizes,
            opacity: cid === -1 ? 0.4 : 0.85,
            line: { width: linewidths, color: linecolors },
          },
        })
      })

      const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(17,24,39,0.8)',
        font: { color: '#9ca3af', size: 11, family: 'Inter, system-ui, sans-serif' },
        margin: { l: 40, r: 20, t: 20, b: 40 },
        xaxis: {
          title: 'UMAP 1',
          gridcolor: 'rgba(55,65,81,0.5)',
          zerolinecolor: 'rgba(55,65,81,0.8)',
          tickfont: { size: 10 },
        },
        yaxis: {
          title: 'UMAP 2',
          gridcolor: 'rgba(55,65,81,0.5)',
          zerolinecolor: 'rgba(55,65,81,0.8)',
          tickfont: { size: 10 },
        },
        legend: {
          bgcolor: 'rgba(17,24,39,0.9)',
          bordercolor: 'rgba(55,65,81,0.8)',
          borderwidth: 1,
          font: { size: 11 },
        },
        hoverlabel: {
          bgcolor: '#1f2937',
          bordercolor: '#374151',
          font: { color: '#e5e7eb', size: 12 },
          align: 'left',
        },
        dragmode: 'zoom',
      }

      const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['select2d', 'lasso2d', 'toImage'],
        displaylogo: false,
        toImageButtonOptions: { format: 'png', filename: 'reqcluster_scatter' },
      }

      // Drop any handler bound on a previous run before re-binding.
      el.removeAllListeners?.('plotly_click')
      Plotly.react(el, traces, layout, config)

      el.on('plotly_click', (data) => {
        if (data.points?.[0]?.customdata) {
          setSelected(data.points[0].customdata)
        }
      })
    }

    loadPlotly()

    return () => { disposed = true }
  }, [loading, requirements, clusters, filterCluster, showNoise, correctionMap])

  // Purge Plotly (frees the WebGL context + listeners) only on unmount.
  useEffect(() => {
    const el = plotRef.current
    return () => {
      if (el) {
        import('plotly.js-dist-min')
          .then(({ default: Plotly }) => Plotly.purge(el))
          .catch(() => {})
      }
    }
  }, [])

  if (loading) return (
    <div className="flex items-center justify-center h-64">
      <Loader size={24} className="animate-spin text-brand-400" />
    </div>
  )

  if (error) return <div className="p-8 text-red-400">{getErrorMessage(error, 'Failed to load data.')}</div>
  if (session?.total_clusters === 0) return <div className="p-8 text-gray-400 text-sm">All requirements were classified as noise (0 clusters). Re-run clustering with a smaller min_cluster_size.</div>

  return (
    <div className="p-6 h-screen flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between flex-shrink-0">
        <div>
          <h1 className="text-xl font-bold text-white">UMAP Scatter Plot</h1>
          <p className="text-sm text-gray-400 mt-0.5">
            {requirements.length} requirements · click a point to inspect
          </p>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={filterCluster}
            onChange={e => setFilterCluster(e.target.value)}
            className="input text-sm py-1.5"
          >
            <option value="all">All Clusters</option>
            {clusters.map(c => (
              <option key={c.cluster_id} value={c.cluster_id}>{c.label}</option>
            ))}
          </select>
          <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
            <input
              type="checkbox"
              checked={showNoise}
              onChange={e => setShowNoise(e.target.checked)}
              className="rounded"
            />
            Show Noise
          </label>
        </div>
      </div>

      {/* Main content */}
      <div className="flex gap-4 flex-1 min-h-0">
        {/* Plot */}
        <div className="flex-1 card overflow-hidden">
          <div ref={plotRef} style={{ width: '100%', height: '100%' }} />
        </div>

        {/* Selected requirement panel */}
        {selected && (
          <div className="w-72 card p-4 flex flex-col gap-3 flex-shrink-0 overflow-y-auto">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-brand-400">{selected.req_id}</span>
              <button onClick={() => setSelected(null)} className="text-gray-600 hover:text-gray-400 text-xs">✕</button>
            </div>
            <p className="text-sm text-gray-200 leading-relaxed">{selected.text}</p>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-500">Cluster</span>
                <span className="flex items-center gap-1.5">
                  <span
                    className="w-2.5 h-2.5 rounded-full"
                    style={{ backgroundColor: getClusterColor(selected.cluster_id) }}
                  />
                  <span className="text-gray-300">{selected.cluster_id === -1 ? 'Noise' : selected.cluster_id}</span>
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Membership</span>
                <span className="text-gray-300">{selected.prob?.toFixed(3) ?? '-'}</span>
              </div>
            </div>

            {selected.correction && (
              <div className={`p-2.5 rounded text-xs border ${
                selected.correction.status === 'pending'
                  ? 'bg-brand-900/20 border-brand-900/50 text-blue-300'
                  : selected.correction.status === 'approved'
                  ? 'bg-emerald-950/20 border-emerald-900/50 text-emerald-300'
                  : 'bg-red-950/20 border-red-900/50 text-red-300'
              }`}>
                <div className="font-semibold uppercase tracking-wider text-[10px] opacity-80 mb-1">
                  Correction: {selected.correction.status}
                </div>
                <p className="italic mb-1">"{selected.correction.comments || 'No comment'}"</p>
                <div className="text-[10px] opacity-60">
                  By: {selected.correction.applied_by} (Conf: {Math.round(selected.correction.confidence * 100)}%)
                </div>
              </div>
            )}

            {selected.cluster_id !== -1 && (
              <button
                onClick={() => navigate(`/s/${sessionId}/clusters/${selected.cluster_id}`)}
                className="btn-secondary text-xs py-1.5 mt-auto"
              >
                View Cluster →
              </button>
            )}
          </div>
        )}
      </div>

      {/* Tip */}
      {!selected && (
        <div className="flex items-center gap-2 text-xs text-gray-600 flex-shrink-0">
          <Info size={12} />
          <span>Click any point to inspect. Scroll to zoom. Drag to pan.</span>
        </div>
      )}
    </div>
  )
}
