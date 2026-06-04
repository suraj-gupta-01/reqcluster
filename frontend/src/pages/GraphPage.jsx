import { useState, useEffect, useRef } from 'react'
import { useParams } from 'react-router-dom'
import { Loader, Info, Sliders } from 'lucide-react'
import { getGraph, getClusters } from '../utils/api.js'
import { getClusterColor } from '../utils/colors.js'

export default function GraphPage() {
  const { sessionId } = useParams()
  const plotRef = useRef(null)

  const [graphData, setGraphData] = useState(null)
  const [clusters, setClusters] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selected, setSelected] = useState(null)
  const [minWeight, setMinWeight] = useState(0.65)
  const [showLabels, setShowLabels] = useState(false)

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const [graph, clus] = await Promise.all([
          getGraph(parseInt(sessionId)),
          getClusters(parseInt(sessionId)),
        ])
        if (!cancelled) { setGraphData(graph); setClusters(clus) }
      } catch {
        if (!cancelled) setError('Failed to load graph data.')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    load()
    return () => { cancelled = true }
  }, [sessionId])

  useEffect(() => {
    if (!graphData || !plotRef.current) return
    const el = plotRef.current
    let disposed = false
    const draw = async () => {
      const Plotly = (await import('plotly.js-dist-min')).default
      if (disposed || !el) return
      const { nodes, edges } = graphData
      const clusterLabelMap = {}
      clusters.forEach(c => { clusterLabelMap[c.cluster_id] = c.label })

      const filteredEdges = edges.filter(e => e.weight >= minWeight)

      // Edge traces
      const edgeX = [], edgeY = []
      filteredEdges.forEach(edge => {
        const src = nodes[edge.source]
        const tgt = nodes[edge.target]
        if (!src || !tgt) return
        edgeX.push(src.x, tgt.x, null)
        edgeY.push(src.y, tgt.y, null)
      })

      const edgeTrace = {
        type: 'scatter', mode: 'lines',
        x: edgeX, y: edgeY,
        line: { width: 0.6, color: 'rgba(20,160,143,0.22)' },
        hoverinfo: 'none', showlegend: false,
      }

      // Node traces grouped by cluster
      const clusterMap = {}
      nodes.forEach(node => {
        const cid = node.cluster_id
        if (!clusterMap[cid]) clusterMap[cid] = []
        clusterMap[cid].push(node)
      })

      const nodeTraces = Object.entries(clusterMap).map(([cidStr, nodeList]) => {
        const cid = parseInt(cidStr)
        const color = getClusterColor(cid)
        const label = cid === -1 ? 'Noise' : (clusterLabelMap[cid] || `Cluster ${cid}`)
        return {
          type: 'scatter',
          mode: showLabels ? 'markers+text' : 'markers',
          name: label,
          x: nodeList.map(n => n.x),
          y: nodeList.map(n => n.y),
          text: showLabels ? nodeList.map(n => n.node_id) : [],
          textposition: 'top center',
          textfont: { size: 8, color: '#9ca3af' },
          customdata: nodeList.map(n => ({
            req_id: n.node_id,
            text: n.requirement_text,
            cluster_id: n.cluster_id,
          })),
          hovertemplate:
            '<b>%{customdata.req_id}</b><br>' +
            '%{customdata.text}<br>' +
            '<i>Cluster: %{customdata.cluster_id}</i><extra></extra>',
          marker: {
            color, size: cid === -1 ? 5 : 8,
            opacity: cid === -1 ? 0.3 : 0.85,
            line: { width: 1, color: 'rgba(0,0,0,0.4)' },
          },
        }
      })

      const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(17,24,39,0.8)',
        font: { color: '#9ca3af', size: 11, family: 'Inter, system-ui, sans-serif' },
        margin: { l: 20, r: 20, t: 20, b: 20 },
        xaxis: { showgrid: false, zeroline: false, showticklabels: false },
        yaxis: { showgrid: false, zeroline: false, showticklabels: false },
        legend: {
          bgcolor: 'rgba(17,24,39,0.9)',
          bordercolor: 'rgba(55,65,81,0.8)',
          borderwidth: 1,
          font: { size: 10 },
          itemsizing: 'constant',
        },
        hoverlabel: {
          bgcolor: '#1f2937', bordercolor: '#374151',
          font: { color: '#e5e7eb', size: 12 }, align: 'left',
        },
        dragmode: 'pan',
      }

      const config = {
        responsive: true, displayModeBar: true,
        modeBarButtonsToRemove: ['select2d', 'lasso2d'],
        displaylogo: false, scrollZoom: true,
      }

      el.removeAllListeners?.('plotly_click')
      Plotly.react(el, [edgeTrace, ...nodeTraces], layout, config)
      el.on('plotly_click', (data) => {
        if (data.points?.[0]?.customdata) setSelected(data.points[0].customdata)
      })
    }
    draw()
    return () => { disposed = true }
  }, [graphData, clusters, minWeight, showLabels])

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
  if (error) return <div className="p-8 text-red-400">{error}</div>

  const nodeCount = graphData?.nodes?.length || 0
  const allEdges = graphData?.edges || []
  const edgeCount = allEdges.filter(e => e.weight >= minWeight).length

  // The backend only computed edges above the clustering threshold, so the
  // slider can't reveal anything below the lowest stored edge weight.
  const minEdgeWeight = allEdges.length ? Math.min(...allEdges.map(e => e.weight)) : 0.5
  const weightFloor = Math.max(0.5, Math.floor(minEdgeWeight * 20) / 20)

  return (
    <div className="p-6 h-screen flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between flex-shrink-0">
        <div>
          <h1 className="text-xl font-bold text-white">Similarity Graph</h1>
          <p className="text-sm text-gray-400 mt-0.5">
            {nodeCount} nodes · {edgeCount} edges (weight ≥ {minWeight})
          </p>
        </div>
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer select-none">
            <input type="checkbox" checked={showLabels}
              onChange={e => setShowLabels(e.target.checked)} className="rounded" />
            Show IDs
          </label>
          <div className="flex items-center gap-2">
            <Sliders size={14} className="text-gray-500" />
            <label className="text-sm text-gray-400">Min Weight</label>
            <input type="range" min={weightFloor} max="0.95" step="0.05"
              value={minWeight}
              onChange={e => setMinWeight(parseFloat(e.target.value))}
              className="w-24 accent-brand-500" />
            <span className="text-sm font-mono text-brand-400 w-10">{minWeight}</span>
          </div>
        </div>
      </div>

      <div className="flex gap-4 flex-1 min-h-0">
        <div className="flex-1 card overflow-hidden">
          <div ref={plotRef} style={{ width: '100%', height: '100%' }} />
        </div>

        {selected && (
          <div className="w-64 card p-4 flex flex-col gap-3 flex-shrink-0 overflow-y-auto">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-brand-400">{selected.req_id}</span>
              <button onClick={() => setSelected(null)}
                className="text-gray-600 hover:text-gray-400 text-xs">✕</button>
            </div>
            <p className="text-sm text-gray-200 leading-relaxed">{selected.text}</p>
            <div className="space-y-1.5 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-500">Cluster</span>
                <span className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: getClusterColor(selected.cluster_id) }} />
                  <span className="text-gray-300">
                    {selected.cluster_id === -1 ? 'Noise' : selected.cluster_id}
                  </span>
                </span>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="flex items-center gap-2 text-xs text-gray-600 flex-shrink-0">
        <Info size={12} />
        <span>Scroll to zoom · drag to pan · click a node to inspect · adjust Min Weight to show more or fewer edges.</span>
      </div>
    </div>
  )
}
