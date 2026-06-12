import { ChevronDown, ChevronUp, GitBranch, ArrowDownToLine, ArrowUpFromLine } from 'lucide-react'
import { useState } from 'react'

const TEXT_LIMIT = 420

// Color map kept in sync with DependencyTreePage RELATION_COLORS
const RELATION_COLORS = {
  data: '#2fbcaa',
  sequential: '#f59e0b',
  hierarchical: '#0ea5e9',
  reference: '#94a3b8',
  semantic: '#22c55e',
}

const truncateText = (value, maxChars = TEXT_LIMIT) => {
  const text = String(value || '').trim()
  if (text.length <= maxChars) return text
  return `${text.slice(0, maxChars).trim()}...`
}

function TextBlock({ title, text }) {
  const [expanded, setExpanded] = useState(false)
  const full = String(text || '').trim()
  const isLong = full.length > TEXT_LIMIT
  const visible = expanded || !isLong ? full : truncateText(full)

  if (!full) return null
  return (
    <div>
      <div className="text-[11px] uppercase tracking-[0.12em] text-gray-500 mb-1">{title}</div>
      <p className="text-sm text-gray-200 leading-relaxed whitespace-normal break-words overflow-hidden">
        {visible}
      </p>
      {isLong && (
        <button
          type="button"
          onClick={() => setExpanded(v => !v)}
          className="mt-1 inline-flex items-center gap-1 text-xs font-semibold text-brand-300 hover:text-brand-200"
        >
          {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}
    </div>
  )
}

function CountPill({ label, value }) {
  return (
    <div className="rounded-lg border border-white/[0.06] bg-white/[0.03] px-3 py-2">
      <div className="text-[10px] uppercase tracking-[0.12em] text-gray-500">{label}</div>
      <div className="font-mono text-sm font-semibold text-gray-100">{value}</div>
    </div>
  )
}

/** Extract the quoted artifact name from a DATA-edge rationale string. */
function extractArtifactName(rationale) {
  const m = String(rationale || '').match(/['"\u2018\u2019\u201c\u201d]([^'"\u2018\u2019\u201c\u201d]+)['"\u2018\u2019\u201c\u201d]/)
  return m ? m[1] : null
}

/** Pill tag for a single artifact name. */
function ArtifactPill({ name }) {
  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold"
      style={{ background: '#2fbcaa22', color: '#2fbcaa', border: '1px solid #2fbcaa44' }}
    >
      {name}
    </span>
  )
}

/**
 * Shows the artifacts this node produces (outgoing DATA edges) and
 * consumes (incoming DATA edges) as labeled pill groups.
 */
function ProducerConsumerSection({ parents, children }) {
  const produced = []
  const consumed = []

  ;(children || []).forEach(item => {
    if (item.relation === 'data') {
      const art = extractArtifactName(item.rationale)
      if (art) produced.push({ art, target: item.node_id })
    }
  })
  ;(parents || []).forEach(item => {
    if (item.relation === 'data') {
      const art = extractArtifactName(item.rationale)
      if (art) consumed.push({ art, source: item.node_id })
    }
  })

  if (!produced.length && !consumed.length) return null

  return (
    <div className="rounded-xl border border-[#2fbcaa]/20 bg-[#2fbcaa]/[0.04] p-3 space-y-2.5">
      <div className="text-[10px] uppercase tracking-[0.14em] text-[#2fbcaa]/70 font-semibold">Data flow</div>
      {produced.length > 0 && (
        <div>
          <div className="flex items-center gap-1.5 mb-1.5">
            <ArrowUpFromLine size={11} className="text-[#2fbcaa]" />
            <span className="text-[10px] text-gray-400 uppercase tracking-wider">Produces</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {produced.map(({ art, target }, i) => (
              <span key={i} title={`→ ${target}`}>
                <ArtifactPill name={art} />
              </span>
            ))}
          </div>
        </div>
      )}
      {consumed.length > 0 && (
        <div>
          <div className="flex items-center gap-1.5 mb-1.5">
            <ArrowDownToLine size={11} className="text-[#2fbcaa]" />
            <span className="text-[10px] text-gray-400 uppercase tracking-wider">Consumes</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {consumed.map(({ art, source }, i) => (
              <span key={i} title={`← ${source}`}>
                <ArtifactPill name={art} />
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function RelatedList({ title, items }) {
  if (!items?.length) return null
  return (
    <div>
      <div className="text-[11px] uppercase tracking-[0.12em] text-gray-500 mb-1">{title}</div>
      <div className="space-y-1.5">
        {items.slice(0, 5).map(item => {
          const color = RELATION_COLORS[item.relation] || '#94a3b8'
          const art = item.relation === 'data' ? extractArtifactName(item.rationale) : null
          return (
            <div key={`${title}-${item.node_id}-${item.relation}`} className="rounded-lg bg-black/20 border border-white/[0.05] px-2.5 py-2">
              <div className="flex items-center justify-between gap-2">
                <span className="font-mono text-xs text-brand-300">{item.node_id}</span>
                <span
                  className="text-[10px] px-1.5 py-0.5 rounded-full font-medium"
                  style={{ background: color + '22', color }}
                >
                  {item.relation}
                </span>
              </div>
              {art && (
                <div className="mt-1.5">
                  <ArtifactPill name={art} />
                </div>
              )}
              <div className="mt-1 text-xs text-gray-400 leading-relaxed whitespace-normal break-words">
                {truncateText(item.requirement_text, 120)}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default function DependencyNodeDetailsPanel({ node }) {
  if (!node) {
    return (
      <div className="h-full min-h-48 flex flex-col items-center justify-center text-center text-gray-500 px-6">
        <GitBranch size={22} className="mb-3 text-gray-600" />
        <p className="text-sm">Hover or click a node to inspect details.</p>
      </div>
    )
  }

  const relationTypes = Object.entries(node.relationCounts || {})
    .filter(([, count]) => count > 0)
    .map(([type, count]) => `${type}: ${count}`)
    .join(', ') || 'None'

  return (
    <div className="space-y-4 max-h-[420px] overflow-y-auto overflow-x-hidden pr-1">
      <div>
        <div className="font-mono text-xs text-brand-300 break-all">{node.node_id}</div>
        <div className="mt-1 text-sm font-semibold text-gray-100 whitespace-normal break-words">
          {node.groupLabel}
        </div>
        <div className="mt-1 text-xs text-gray-500">
          Depth {node.level} · Cluster {node.cluster_id === -1 ? 'Noise' : node.cluster_id}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <CountPill label="Parents" value={node.parentCount} />
        <CountPill label="Children" value={node.childCount} />
        <CountPill label="Incoming" value={node.incomingCount} />
        <CountPill label="Outgoing" value={node.outgoingCount} />
      </div>

      <div>
        <div className="text-[11px] uppercase tracking-[0.12em] text-gray-500 mb-1">Dependency types</div>
        <div className="text-xs text-gray-300 whitespace-normal break-words">{relationTypes}</div>
      </div>

      <TextBlock title="Requirement text" text={node.requirement_text} />
      <TextBlock title="Rationale" text={node.rationale} />

      <ProducerConsumerSection parents={node.parents} children={node.children} />

      <RelatedList title="Parent preview" items={node.parents} />
      <RelatedList title="Child preview" items={node.children} />
    </div>
  )
}
