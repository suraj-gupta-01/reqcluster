import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import {
  Workflow, Loader, RefreshCw, AlertTriangle, GitBranch, ArrowRight,
  Search, SlidersHorizontal, X, MousePointer2,
} from 'lucide-react'
import {
  getSessions, generateDependencies, getDependencies, getErrorMessage,
} from '../utils/api.js'
import { getClusterColor } from '../utils/colors.js'
import DependencyNodeDetailsPanel from '../components/DependencyNodeDetailsPanel.jsx'

const RELATION_COLORS = {
  data: '#2fbcaa',
  sequential: '#f59e0b',
  hierarchical: '#0ea5e9',
  reference: '#94a3b8',
  semantic: '#22c55e',
}
const RELATION_LABEL = {
  data: 'Data (producer → consumer)',
  sequential: 'Sequential (precondition)',
  hierarchical: 'Hierarchical',
  reference: 'Explicit reference',
  semantic: 'Semantic',
}
const LARGE_NODE_THRESHOLD = 250
const LARGE_EDGE_THRESHOLD = 600
const DEFAULT_LARGE_NODE_LIMIT = 250
const DEFAULT_LARGE_EDGE_LIMIT = 400
const DEFAULT_LARGE_DEPTH = 8
const SAFE_NODE_CAP = 1000
const SAFE_EDGE_CAP = 3000

function truncateText(value, maxChars = 120) {
  const text = String(value || '').replace(/\s+/g, ' ').trim()
  if (text.length <= maxChars) return text
  return `${text.slice(0, maxChars).trim()}...`
}

function buildShortHoverLabel(node) {
  return [
    `<b>${node.node_id}</b>`,
    `Group: ${truncateText(node.groupLabel, 36)}`,
    `Depth ${node.level} · Degree ${node.degree}`,
    'Click for details',
  ].join('<br>')
}

function getAvailableGroups(nodes, groupMap = {}) {
  const groups = new Map()
  nodes.forEach(node => {
    const label = groupMap[node.cluster_id]?.label || (node.cluster_id === -1 ? 'Noise' : `Cluster ${node.cluster_id}`)
    groups.set(String(node.cluster_id), { value: String(node.cluster_id), label })
  })
  return [...groups.values()].sort((a, b) => a.label.localeCompare(b.label))
}

function getAvailableEdgeTypes(edges) {
  return [...new Set(edges.map(edge => edge.relation || 'semantic'))].sort()
}

function computeNodeDegrees(edges) {
  const degrees = {}
  edges.forEach(edge => {
    degrees[edge.source] = (degrees[edge.source] || 0) + 1
    degrees[edge.target] = (degrees[edge.target] || 0) + 1
  })
  return degrees
}

function getNodeNeighborhood(nodeId, edges, hopDepth = 1) {
  const id = Number(nodeId)
  if (!Number.isFinite(id) || hopDepth <= 0) return new Set()
  const result = new Set([id])
  let frontier = new Set([id])
  for (let hop = 0; hop < hopDepth; hop += 1) {
    const next = new Set()
    edges.forEach(edge => {
      if (frontier.has(edge.source)) next.add(edge.target)
      if (frontier.has(edge.target)) next.add(edge.source)
    })
    next.forEach(value => result.add(value))
    frontier = next
  }
  return result
}

function buildGroupMap(rationale) {
  const map = {}
  ;(rationale?.grouping || []).forEach(group => {
    map[group.cluster_id] = group
  })
  return map
}

function buildEdgeIndexes(nodes, edges, groupMap, dependencyRationale) {
  const byId = new Map(nodes.map(node => [node.id, node]))
  const depsByKey = new Map()
  dependencyRationale.forEach(dep => depsByKey.set(`${dep.source}-${dep.target}-${dep.relation}`, dep))

  const incoming = {}
  const outgoing = {}
  const relationCounts = {}
  edges.forEach(edge => {
    const source = byId.get(edge.source)
    const target = byId.get(edge.target)
    if (!source || !target) return
    const dep = depsByKey.get(`${edge.source}-${edge.target}-${edge.relation}`) || {}
    const out = {
      ...target,
      relation: edge.relation,
      weight: edge.weight,
      rationale: dep.justification || edge.rationale || '',
    }
    const inc = {
      ...source,
      relation: edge.relation,
      weight: edge.weight,
      rationale: dep.justification || edge.rationale || '',
    }
    ;(outgoing[edge.source] ||= []).push(out)
    ;(incoming[edge.target] ||= []).push(inc)
    relationCounts[edge.source] ||= {}
    relationCounts[edge.target] ||= {}
    relationCounts[edge.source][edge.relation] = (relationCounts[edge.source][edge.relation] || 0) + 1
    relationCounts[edge.target][edge.relation] = (relationCounts[edge.target][edge.relation] || 0) + 1
  })

  return { incoming, outgoing, relationCounts, groupMap }
}

function enrichNode(node, indexes, degrees) {
  if (!node) return null
  const group = indexes.groupMap[node.cluster_id]
  const parents = indexes.incoming[node.id] || []
  const children = indexes.outgoing[node.id] || []
  return {
    ...node,
    groupLabel: group?.label || (node.cluster_id === -1 ? 'Noise' : `Cluster ${node.cluster_id}`),
    degree: degrees[node.id] || 0,
    parentCount: parents.length,
    childCount: children.length,
    incomingCount: parents.length,
    outgoingCount: children.length,
    relationCounts: indexes.relationCounts[node.id] || {},
    rationale: group?.rationale || '',
    parents,
    children,
  }
}

function edgeSort(a, b) {
  const weightDelta = Number(b.weight || 0) - Number(a.weight || 0)
  if (weightDelta !== 0) return weightDelta
  if (a.source !== b.source) return a.source - b.source
  return a.target - b.target
}

function nodeSort(a, b, degrees, matchedIds, focusIds, selectedGroup) {
  const aMatch = matchedIds.has(a.id) ? 1 : 0
  const bMatch = matchedIds.has(b.id) ? 1 : 0
  if (aMatch !== bMatch) return bMatch - aMatch
  const aFocus = focusIds.has(a.id) ? 1 : 0
  const bFocus = focusIds.has(b.id) ? 1 : 0
  if (aFocus !== bFocus) return bFocus - aFocus
  const aGroup = selectedGroup !== 'all' && String(a.cluster_id) === selectedGroup ? 1 : 0
  const bGroup = selectedGroup !== 'all' && String(b.cluster_id) === selectedGroup ? 1 : 0
  if (aGroup !== bGroup) return bGroup - aGroup
  const degreeDelta = (degrees[b.id] || 0) - (degrees[a.id] || 0)
  if (degreeDelta !== 0) return degreeDelta
  if (a.level !== b.level) return a.level - b.level
  return String(a.node_id).localeCompare(String(b.node_id))
}

function filterNodesAndEdges(rawNodes, rawEdges, filters, groupMap = {}) {
  const typedEdges = rawEdges
    .map(edge => ({ ...edge, relation: edge.relation || 'semantic', weight: Number(edge.weight ?? 0) }))
    .filter(edge => filters.edgeTypes[edge.relation] !== false)
    .filter(edge => edge.weight >= filters.minStrength)
    .sort(edgeSort)

  const nodeById = new Map(rawNodes.map(node => [node.id, node]))
  const degrees = computeNodeDegrees(typedEdges)
  const query = filters.search.trim().toLowerCase()
  const matchedIds = new Set()
  const focusIds = filters.focusNodeId && filters.focusDepth !== 'none'
    ? getNodeNeighborhood(filters.focusNodeId, typedEdges, Number(filters.focusDepth))
    : new Set()

  let candidateIds = new Set(rawNodes.map(node => node.id))

  if (!filters.showFullDepth) {
    candidateIds = new Set([...candidateIds].filter(id => {
      const level = Number(nodeById.get(id)?.level ?? 0)
      return level >= filters.minDepth && level <= filters.maxDepth
    }))
  }

  if (filters.group !== 'all') {
    const groupIds = new Set(rawNodes.filter(node => String(node.cluster_id) === filters.group).map(node => node.id))
    const contextIds = new Set(groupIds)
    typedEdges.forEach(edge => {
      if (groupIds.has(edge.source)) contextIds.add(edge.target)
      if (groupIds.has(edge.target)) contextIds.add(edge.source)
    })
    candidateIds = new Set([...candidateIds].filter(id => contextIds.has(id)))
  }

  if (query) {
    rawNodes.forEach(node => {
      const label = groupMap[node.cluster_id]?.label || ''
      const haystack = `${node.node_id} ${node.requirement_text} ${label}`.toLowerCase()
      if (haystack.includes(query)) matchedIds.add(node.id)
    })
    const searchContext = new Set(matchedIds)
    typedEdges.forEach(edge => {
      if (matchedIds.has(edge.source)) searchContext.add(edge.target)
      if (matchedIds.has(edge.target)) searchContext.add(edge.source)
    })
    candidateIds = new Set([...candidateIds].filter(id => searchContext.has(id)))
  }

  if (focusIds.size) {
    candidateIds = new Set([...candidateIds].filter(id => focusIds.has(id)))
  }

  const sortedNodes = rawNodes
    .filter(node => candidateIds.has(node.id))
    .sort((a, b) => nodeSort(a, b, degrees, matchedIds, focusIds, filters.group))

  const limitedNodes = sortedNodes.slice(0, Math.min(filters.nodeLimit, SAFE_NODE_CAP))
  const visibleIds = new Set(limitedNodes.map(node => node.id))
  const limitedEdges = typedEdges
    .filter(edge => visibleIds.has(edge.source) && visibleIds.has(edge.target))
    .slice(0, Math.min(filters.edgeLimit, SAFE_EDGE_CAP))

  return {
    nodes: limitedNodes,
    edges: limitedEdges,
    degrees,
    matchedIds,
    focusIds,
    totalCandidateNodes: sortedNodes.length,
    totalCandidateEdges: typedEdges.filter(edge => candidateIds.has(edge.source) && candidateIds.has(edge.target)).length,
  }
}

function buildDependencyGraphViewModel(rawTreeData, filters) {
  const rawNodes = rawTreeData?.nodes || []
  const rawEdges = rawTreeData?.edges || []
  const groupMap = buildGroupMap(rawTreeData?.rationale)
  const dependencyRationale = rawTreeData?.rationale?.dependencies || []
  const filtered = filterNodesAndEdges(rawNodes, rawEdges, filters, groupMap)
  const indexes = buildEdgeIndexes(rawNodes, rawEdges, groupMap, dependencyRationale)
  return {
    ...filtered,
    groupMap,
    nodeDetails: new Map(rawNodes.map(node => [node.id, enrichNode(node, indexes, filtered.degrees)])),
  }
}

function defaultFiltersForData(data) {
  const nodes = data?.nodes || []
  const edges = data?.edges || []
  const large = nodes.length > LARGE_NODE_THRESHOLD || edges.length > LARGE_EDGE_THRESHOLD
  const maxDepth = Number(data?.stats?.max_depth ?? Math.max(0, ...nodes.map(n => Number(n.level || 0))))
  const edgeTypes = {}
  getAvailableEdgeTypes(edges).forEach(type => { edgeTypes[type] = true })
  return {
    group: 'all',
    search: '',
    minDepth: 0,
    maxDepth: large ? Math.min(DEFAULT_LARGE_DEPTH, maxDepth) : maxDepth,
    showFullDepth: !large,
    edgeTypes,
    edgeLimit: large ? Math.min(DEFAULT_LARGE_EDGE_LIMIT, edges.length) : Math.min(edges.length, SAFE_EDGE_CAP),
    nodeLimit: large ? Math.min(DEFAULT_LARGE_NODE_LIMIT, nodes.length) : Math.min(nodes.length, SAFE_NODE_CAP),
    minStrength: 0,
    focusDepth: large ? '1' : 'none',
    focusNodeId: null,
  }
}

function ControlLabel({ children }) {
  return <label className="text-[11px] uppercase tracking-[0.12em] text-gray-500 mb-1 block">{children}</label>
}

export default function DependencyTreePage() {
  const plotRef = useRef(null)
  const [sessions, setSessions] = useState([])
  const [selected, setSelected] = useState(null)
  const [data, setData] = useState(null)
  const [loadingSessions, setLoadingSessions] = useState(true)
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState(null)
  const [inspectedNodeId, setInspectedNodeId] = useState(null)
  const [viewMode, setViewMode] = useState('3D')
  const [filters, setFilters] = useState(defaultFiltersForData(null))

  useEffect(() => {
    getSessions()
      .then(list => {
        const done = list.filter(s => s.status === 'done')
        setSessions(done)
        if (done.length && !selected) setSelected(done[0].id)
      })
      .catch(() => setError('Failed to load sessions.'))
      .finally(() => setLoadingSessions(false))
  }, [selected])

  const load = useCallback(async (sid) => {
    try {
      const res = await getDependencies(sid)
      setData(res)
      setFilters(defaultFiltersForData(res))
      setInspectedNodeId(null)
      setError(null)
    } catch {
      setData(null)
      setInspectedNodeId(null)
    }
  }, [])

  useEffect(() => { if (selected) load(selected) }, [selected, load])

  const handleGenerate = async () => {
    if (!selected) return
    setGenerating(true)
    setError(null)
    try {
      const res = await generateDependencies({ session_id: selected })
      setData(res)
      setFilters(defaultFiltersForData(res))
      setInspectedNodeId(null)
    } catch (err) {
      setError(getErrorMessage(err, 'Failed to generate dependency tree.'))
    } finally {
      setGenerating(false)
    }
  }

  const stats = data?.stats || {}
  const rawNodes = useMemo(() => data?.nodes || [], [data])
  const rawEdges = useMemo(() => data?.edges || [], [data])
  const isLargeGraph = rawNodes.length > LARGE_NODE_THRESHOLD || rawEdges.length > LARGE_EDGE_THRESHOLD
  const groups = useMemo(() => getAvailableGroups(rawNodes, buildGroupMap(data?.rationale)), [rawNodes, data])
  const edgeTypes = useMemo(() => getAvailableEdgeTypes(rawEdges), [rawEdges])
  const maxDepthAvailable = Number(stats.max_depth ?? Math.max(0, ...rawNodes.map(n => Number(n.level || 0))))
  const viewModel = useMemo(
    () => (data ? buildDependencyGraphViewModel(data, filters) : null),
    [data, filters],
  )
  const inspectedNode = inspectedNodeId !== null ? viewModel?.nodeDetails.get(inspectedNodeId) : null
  const grouping = data?.rationale?.grouping || []

  const updateFilter = (patch) => setFilters(current => ({ ...current, ...patch }))
  const resetFilters = () => {
    const next = defaultFiltersForData(data)
    setFilters(next)
    setInspectedNodeId(null)
  }

  useEffect(() => {
    if (!data || !plotRef.current || !viewModel) return
    const el = plotRef.current
    let disposed = false
    const draw = async () => {
      const Plotly = (await import('plotly.js-dist-min')).default
      if (disposed || !el) return
      const { nodes, edges, matchedIds, focusIds, degrees, groupMap } = viewModel
      const is3D = viewMode === '3D'
      const dense = rawNodes.length > LARGE_NODE_THRESHOLD || rawEdges.length > LARGE_EDGE_THRESHOLD
      const byLevel = {}
      nodes.forEach(n => { (byLevel[n.level] ||= []).push(n) })
      const pos = {}
      Object.entries(byLevel).forEach(([lvl, group]) => {
        const sorted = [...group].sort((a, b) => String(a.node_id).localeCompare(String(b.node_id)))
        const L = sorted.length
        sorted.forEach((n, i) => {
          if (is3D) {
            if (L === 1) {
              pos[n.id] = { x: Number(lvl), y: 0, z: 0 }
            } else {
              const radius = 0.75 * Math.sqrt(L)
              const angle = (2 * Math.PI * i) / L
              pos[n.id] = { x: Number(lvl), y: radius * Math.cos(angle), z: radius * Math.sin(angle) }
            }
          } else {
            const clusterOffset = Number(n.cluster_id === -1 ? 0 : n.cluster_id) * 0.18
            pos[n.id] = { x: Number(lvl), y: i - (L - 1) / 2 + clusterOffset, z: 0 }
          }
        })
      })

      const edgeTraces = edgeTypes.map(rel => {
        const ex = []
        const ey = []
        const ez = []
        edges.filter(e => e.relation === rel).forEach(e => {
          const s = pos[e.source]
          const t = pos[e.target]
          if (!s || !t) return
          ex.push(s.x, t.x, null)
          ey.push(s.y, t.y, null)
          ez.push(s.z, t.z, null)
        })
        return {
          type: is3D ? 'scatter3d' : 'scatter',
          mode: 'lines',
          x: ex,
          y: ey,
          ...(is3D ? { z: ez } : {}),
          name: RELATION_LABEL[rel] || rel,
          line: is3D
            ? { width: dense ? 1.2 : 2.2, color: RELATION_COLORS[rel] || '#94a3b8' }
            : { width: dense ? 0.8 : 1.2, color: RELATION_COLORS[rel] || '#94a3b8', shape: 'spline' },
          hoverinfo: 'none',
          opacity: dense ? 0.34 : 0.68,
        }
      })

      const markerSizes = nodes.map(node => {
        if (node.id === inspectedNodeId) return is3D ? 10 : 16
        if (matchedIds.has(node.id)) return is3D ? 9 : 15
        if (focusIds.has(node.id)) return is3D ? 8 : 13
        return dense ? (is3D ? 4.2 : 7) : (is3D ? 7 : 11)
      })
      const markerLines = nodes.map(node => {
        if (node.id === inspectedNodeId) return '#f8fafc'
        if (matchedIds.has(node.id)) return '#f59e0b'
        if (focusIds.has(node.id)) return '#2fbcaa'
        return 'rgba(0,0,0,0.5)'
      })
      const enrichedNodes = nodes.map(node => ({
        ...node,
        groupLabel: groupMap[node.cluster_id]?.label || (node.cluster_id === -1 ? 'Noise' : `Cluster ${node.cluster_id}`),
        degree: degrees[node.id] || 0,
      }))

      const nodeTrace = {
        type: is3D ? 'scatter3d' : 'scatter',
        mode: 'markers',
        name: 'Requirements',
        showlegend: false,
        x: nodes.map(n => pos[n.id].x),
        y: nodes.map(n => pos[n.id].y),
        ...(is3D ? { z: nodes.map(n => pos[n.id].z) } : {}),
        text: enrichedNodes.map(buildShortHoverLabel),
        customdata: enrichedNodes.map(n => [n.id]),
        hovertemplate: '%{text}<extra></extra>',
        marker: {
          size: markerSizes,
          color: nodes.map(n => getClusterColor(n.cluster_id)),
          opacity: dense ? 0.82 : 0.95,
          line: { width: nodes.map(n => (n.id === inspectedNodeId || matchedIds.has(n.id) ? 2.4 : 0.8)), color: markerLines },
        },
      }

      const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#9ca3af', size: 11, family: 'Plus Jakarta Sans, sans-serif' },
        margin: is3D ? { l: 0, r: 0, t: 10, b: 0 } : { l: 35, r: 18, t: 16, b: 34 },
        uirevision: `dependency-tree-${selected}-${viewMode}`,
        ...(is3D ? {
          scene: {
            xaxis: { title: 'Depth', gridcolor: 'rgba(255,255,255,0.06)', backgroundcolor: 'rgba(0,0,0,0)', showbackground: false, zeroline: false },
            yaxis: { title: '', showgrid: false, showbackground: false, showticklabels: false, zeroline: false },
            zaxis: { title: '', showgrid: false, showbackground: false, showticklabels: false, zeroline: false },
            camera: { eye: { x: 1.65, y: 1.55, z: 1.2 } },
          },
        } : {
          xaxis: { title: 'Dependency depth', gridcolor: 'rgba(255,255,255,0.06)', zeroline: false },
          yaxis: { showgrid: false, zeroline: false, showticklabels: false },
        }),
        legend: {
          bgcolor: 'rgba(17,22,22,0.78)',
          bordercolor: 'rgba(255,255,255,0.08)',
          borderwidth: 1,
          orientation: 'h',
          y: 1.08,
          font: { size: 10 },
        },
        hoverlabel: {
          bgcolor: '#11161a',
          bordercolor: '#24303a',
          font: { color: '#e5e7eb', size: 12 },
          align: 'left',
          namelength: 32,
        },
        dragmode: is3D ? 'orbit' : 'pan',
      }
      const config = {
        responsive: true,
        displaylogo: false,
        scrollZoom: true,
        ...(is3D ? {} : { modeBarButtonsToRemove: ['select2d', 'lasso2d'] }),
      }
      el.removeAllListeners?.('plotly_click')
      el.removeAllListeners?.('plotly_hover')
      Plotly.react(el, [...edgeTraces, nodeTrace], layout, config)
      el.on('plotly_hover', ev => {
        const id = ev.points?.[0]?.customdata?.[0]
        if (id !== undefined) setInspectedNodeId(Number(id))
      })
      el.on('plotly_click', ev => {
        const id = ev.points?.[0]?.customdata?.[0]
        if (id === undefined) return
        setInspectedNodeId(Number(id))
        setFilters(current => current.focusDepth === 'none'
          ? current
          : { ...current, focusNodeId: Number(id) })
      })
    }
    draw()
    return () => { disposed = true }
  }, [data, viewModel, viewMode, selected, edgeTypes, rawNodes.length, rawEdges.length, inspectedNodeId])

  useEffect(() => {
    const el = plotRef.current
    return () => { if (el) import('plotly.js-dist-min').then(({ default: P }) => P.purge(el)).catch(() => {}) }
  }, [])

  return (
    <div className="p-6 lg:p-8 max-w-[1600px] animate-fade-up overflow-x-hidden">
      <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Workflow size={20} className="text-brand-400" />
            <h1 className="text-2xl font-bold text-white tracking-tight">Dependency Tree</h1>
          </div>
          <p className="text-gray-400 text-sm max-w-2xl">
            Relationships inferred from named artifact data-flow (producer→consumer),
            explicit requirement cross-references, temporal preconditions, and cluster hierarchy.
          </p>
        </div>
        <button onClick={handleGenerate} disabled={!selected || generating} className="btn-primary text-sm">
          {generating ? <><Loader size={14} className="animate-spin" /> Analyzing...</> : <><RefreshCw size={14} /> Build dependency tree</>}
        </button>
      </div>

      <div className="card p-4 mb-4">
        <ControlLabel>Session</ControlLabel>
        {loadingSessions ? (
          <div className="flex items-center gap-2 text-gray-400 text-sm"><Loader size={14} className="animate-spin" /> Loading...</div>
        ) : sessions.length === 0 ? (
          <div className="text-sm text-gray-500">No completed sessions. Run clustering first.</div>
        ) : (
          <select value={selected || ''} onChange={e => setSelected(parseInt(e.target.value))} className="input text-sm w-full md:w-96">
            {sessions.map(s => <option key={s.id} value={s.id}>#{s.id} - {s.filename} ({s.total_requirements} reqs)</option>)}
          </select>
        )}
      </div>

      {error && (
        <div className="card p-3 mb-4 border-amber-900/30 flex items-start gap-2">
          <AlertTriangle size={14} className="text-amber-400 mt-0.5 flex-shrink-0" />
          <p className="text-sm text-amber-300/80 break-words">{error}</p>
        </div>
      )}

      {!data ? (
        <div className="card p-12 text-center text-gray-500 text-sm">
          <GitBranch size={28} className="mx-auto mb-3 text-gray-600" />
          No dependency tree yet. Click <span className="text-brand-400 font-medium">Build dependency tree</span> to analyze this session.
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-4">
            {[
              ['Requirements', stats.n_nodes],
              ['Dependencies', stats.n_edges],
              ['Max depth', stats.max_depth],
              ['Roots', stats.root_count],
            ].map(([label, val]) => (
              <div key={label} className="stat-card">
                <span className="text-xs text-gray-500 uppercase tracking-wider">{label}</span>
                <span className="text-2xl font-bold text-white font-mono">{val ?? '-'}</span>
              </div>
            ))}
          </div>

          {isLargeGraph && (
            <div className="card p-3 mb-4 border-brand-900/30 flex items-start gap-2">
              <SlidersHorizontal size={14} className="text-brand-300 mt-0.5 flex-shrink-0" />
              <p className="text-sm text-brand-100/80">
                Large graph simplified for readability. Use filters or focus mode to explore more.
              </p>
            </div>
          )}
          {(rawNodes.length > SAFE_NODE_CAP || rawEdges.length > SAFE_EDGE_CAP) && (
            <div className="card p-3 mb-4 border-amber-900/30 flex items-start gap-2">
              <AlertTriangle size={14} className="text-amber-400 mt-0.5 flex-shrink-0" />
              <p className="text-sm text-amber-300/80">
                This graph is large. Use filters before rendering full graph.
              </p>
            </div>
          )}

          <div className="card p-4 mb-4">
            <div className="flex items-center gap-2 mb-3">
              <SlidersHorizontal size={15} className="text-brand-400" />
              <h2 className="text-sm font-semibold text-gray-200">Graph Controls</h2>
            </div>
            <div className="grid gap-4 lg:grid-cols-6 md:grid-cols-3">
              <div>
                <ControlLabel>View</ControlLabel>
                <div className="flex rounded-lg bg-gray-900 border border-white/[0.08] p-0.5">
                  {['3D', '2D'].map(mode => (
                    <button
                      key={mode}
                      type="button"
                      onClick={() => setViewMode(mode)}
                      className={`flex-1 px-3 py-1.5 text-xs font-semibold rounded-md transition-all ${
                        viewMode === mode ? 'bg-brand-500/15 text-brand-300 border border-brand-500/20' : 'text-gray-400 border border-transparent'
                      }`}
                    >
                      {mode}
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <ControlLabel>Group</ControlLabel>
                <select value={filters.group} onChange={e => updateFilter({ group: e.target.value, focusNodeId: null })} className="input text-sm w-full">
                  <option value="all">All groups</option>
                  {groups.map(group => <option key={group.value} value={group.value}>{group.label}</option>)}
                </select>
              </div>
              <div className="md:col-span-2">
                <ControlLabel>Search</ControlLabel>
                <div className="relative">
                  <Search size={14} className="absolute left-3 top-2.5 text-gray-500" />
                  <input
                    value={filters.search}
                    onChange={e => updateFilter({ search: e.target.value, focusNodeId: null })}
                    className="input text-sm w-full pl-9"
                    placeholder="Requirement ID or text"
                  />
                </div>
              </div>
              <div>
                <ControlLabel>Focus depth</ControlLabel>
                <select value={filters.focusDepth} onChange={e => updateFilter({ focusDepth: e.target.value })} className="input text-sm w-full">
                  <option value="none">None</option>
                  <option value="1">1-hop</option>
                  <option value="2">2-hop</option>
                </select>
              </div>
              <div className="flex items-end gap-2">
                <button type="button" onClick={resetFilters} className="btn-secondary text-sm w-full">
                  <X size={14} /> Reset
                </button>
              </div>
            </div>

            <div className="grid gap-4 lg:grid-cols-4 md:grid-cols-2 mt-4">
              <div>
                <ControlLabel>Max depth {filters.showFullDepth ? '(full)' : filters.maxDepth}</ControlLabel>
                <input
                  type="range"
                  min="0"
                  max={Math.max(maxDepthAvailable, 0)}
                  value={filters.showFullDepth ? maxDepthAvailable : filters.maxDepth}
                  disabled={filters.showFullDepth}
                  onChange={e => updateFilter({ maxDepth: Number(e.target.value) })}
                  className="w-full accent-brand-500"
                />
                <label className="mt-1 flex items-center gap-2 text-xs text-gray-400">
                  <input
                    type="checkbox"
                    checked={filters.showFullDepth}
                    onChange={e => updateFilter({ showFullDepth: e.target.checked })}
                    className="accent-brand-500"
                  />
                  Show full depth
                </label>
              </div>
              <div>
                <ControlLabel>Node limit {filters.nodeLimit}</ControlLabel>
                <input
                  type="range"
                  min="25"
                  max={Math.min(Math.max(rawNodes.length, 25), SAFE_NODE_CAP)}
                  step="25"
                  value={filters.nodeLimit}
                  onChange={e => updateFilter({ nodeLimit: Number(e.target.value) })}
                  className="w-full accent-brand-500"
                />
              </div>
              <div>
                <ControlLabel>Edge limit {filters.edgeLimit}</ControlLabel>
                <input
                  type="range"
                  min="25"
                  max={Math.min(Math.max(rawEdges.length, 25), SAFE_EDGE_CAP)}
                  step="25"
                  value={filters.edgeLimit}
                  onChange={e => updateFilter({ edgeLimit: Number(e.target.value) })}
                  className="w-full accent-brand-500"
                />
              </div>
              <div>
                <ControlLabel>Min strength {filters.minStrength.toFixed(2)}</ControlLabel>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={filters.minStrength}
                  onChange={e => updateFilter({ minStrength: Number(e.target.value) })}
                  className="w-full accent-brand-500"
                />
              </div>
            </div>

            <div className="mt-4 flex flex-wrap gap-3">
              {edgeTypes.map(type => (
                <label key={type} className="inline-flex items-center gap-2 text-xs text-gray-300">
                  <input
                    type="checkbox"
                    checked={filters.edgeTypes[type] !== false}
                    onChange={e => updateFilter({ edgeTypes: { ...filters.edgeTypes, [type]: e.target.checked } })}
                    className="accent-brand-500"
                  />
                  <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ backgroundColor: RELATION_COLORS[type] || '#94a3b8' }} />
                  {RELATION_LABEL[type] || type}
                </label>
              ))}
            </div>
          </div>

          <div className="mb-4 flex flex-wrap items-center justify-between gap-3 text-xs text-gray-400">
            <div>
              Showing <span className="font-mono text-gray-100">{viewModel?.nodes.length || 0}</span> of <span className="font-mono text-gray-100">{rawNodes.length}</span> nodes and{' '}
              <span className="font-mono text-gray-100">{viewModel?.edges.length || 0}</span> of <span className="font-mono text-gray-100">{rawEdges.length}</span> dependencies
              {viewModel && viewModel.totalCandidateNodes !== rawNodes.length && (
                <span> ({viewModel.totalCandidateNodes} candidate nodes after filters)</span>
              )}
            </div>
            {filters.focusNodeId !== null && (
              <button type="button" onClick={() => updateFilter({ focusNodeId: null })} className="btn-secondary text-xs py-1.5 px-3">
                <MousePointer2 size={13} /> Clear focus
              </button>
            )}
          </div>

          <div className="grid xl:grid-cols-[minmax(0,1fr)_380px] gap-4">
            <div className="card overflow-hidden min-w-0" style={{ height: 560 }}>
              <div ref={plotRef} style={{ width: '100%', height: '100%' }} />
            </div>

            <div className="space-y-4 min-w-0">
              <div className="card p-4 overflow-hidden">
                <h2 className="text-sm font-semibold text-gray-300 mb-3">Node details</h2>
                <DependencyNodeDetailsPanel node={inspectedNode} />
              </div>

              <div className="card p-4 overflow-hidden">
                <h2 className="text-sm font-semibold text-gray-300 mb-3">Grouping rationale</h2>
                <div className="space-y-3 max-h-[360px] overflow-y-auto overflow-x-hidden pr-1">
                  {grouping.map(group => (
                    <div key={group.cluster_id} className="border-l-2 pl-3" style={{ borderColor: getClusterColor(group.cluster_id) }}>
                      <div className="text-sm font-medium text-gray-200 whitespace-normal break-words">{group.label}</div>
                      <p className="text-xs text-gray-400 mt-0.5 leading-relaxed whitespace-normal break-words">
                        {group.rationale}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {data?.rationale?.dependencies?.length > 0 && (
            <div className="card mt-4 overflow-hidden">
              <div className="px-4 py-3 border-b border-white/[0.06] text-sm font-semibold text-gray-300">
                Dependency justifications
              </div>
              <div className="divide-y divide-white/[0.04] max-h-72 overflow-y-auto overflow-x-hidden">
                {data.rationale.dependencies.slice(0, 100).map((dep, i) => {
                  const color = RELATION_COLORS[dep.relation] || '#94a3b8'
                  // Extract quoted artifact name from DATA-edge rationale
                  const artifactMatch = dep.relation === 'data'
                    ? String(dep.justification || '').match(/['‘’“”"]([^'‘’“”"]+)['‘’“”"]/)
                    : null
                  const artifactName = artifactMatch ? artifactMatch[1] : null
                  return (
                    <div key={`${dep.source}-${dep.target}-${dep.relation}-${i}`} className="px-4 py-2.5 flex items-start gap-3 text-sm min-w-0">
                      <span className="font-mono text-xs text-gray-400 flex items-center gap-1.5 flex-shrink-0">
                        {dep.source_req_id}<ArrowRight size={12} className="text-gray-600" />{dep.target_req_id}
                      </span>
                      <span
                        className="badge text-[10px] flex-shrink-0 px-1.5 py-0.5 rounded-full font-medium"
                        style={{ backgroundColor: color + '22', color }}
                      >
                        {dep.relation}
                      </span>
                      {artifactName && (
                        <span
                          className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold flex-shrink-0"
                          style={{ background: '#2fbcaa22', color: '#2fbcaa', border: '1px solid #2fbcaa44' }}
                        >
                          {artifactName}
                        </span>
                      )}
                      <span className="text-xs text-gray-500 whitespace-normal break-words min-w-0">{dep.justification}</span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
