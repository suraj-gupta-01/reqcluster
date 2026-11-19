import { ChevronDown, ChevronUp, GitBranch, ArrowDownToLine, ArrowUpFromLine, Sparkles } from 'lucide-react'
import { useState } from 'react'
import { explainEdge } from '../api/endpoints.js'
import { getErrorMessage } from '../utils/format.js'

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

/**
 * Artifact name for a DATA edge.
 *
 * Prefers the structured evidence the engine now emits; the rationale-string
 * regex is only a fallback for dependency trees persisted before evidence
 * existed, so old sessions keep rendering instead of going blank.
 */
function extractArtifactName(item) {
  const evidence = item?.evidence || {}
  const structured = evidence.producer_artifact_name || evidence.artifact
  if (structured) return structured
  const m = String(item?.rationale || '').match(/['"\u2018\u2019\u201c\u201d]([^'"\u2018\u2019\u201c\u201d]+)['"\u2018\u2019\u201c\u201d]/)
  return m ? m[1] : null
}

const TIER_COLORS = {
  High: '#22c55e',
  Medium: '#f59e0b',
  Low: '#94a3b8',
}

/** Coarse confidence bucket, shown instead of a raw weight. */
function ConfidenceBadge({ tier }) {
  if (!tier) return null
  const color = TIER_COLORS[tier] || '#94a3b8'
  return (
    <span
      className="text-[10px] px-1.5 py-0.5 rounded-full font-medium"
      style={{ background: `${color}22`, color }}
      title="Confidence tier derived from the edge weight"
    >
      {tier}
    </span>
  )
}

/** Human-readable label for each inference pass. */
const PASS_LABELS = {
  pass_0_artifact_match: 'Shared artifact (producer \u2192 consumer)',
  pass_1_explicit_reference: 'Explicit requirement reference',
  pass_2_sequential: 'Sequential cue',
  pass_2_hierarchical: 'Same cluster, broader wording',
}

const EVIDENCE_FIELD_LABELS = {
  artifact: 'Artifact',
  producer_artifact_name: 'Produced as',
  consumer_artifact_name: 'Consumed as',
  artifact_similarity: 'Synonym similarity',
  matched_text: 'Matched text',
  hierarchical_cue: 'Hierarchical cue',
  cue: 'Sequential cue',
  cosine_similarity: 'Similarity',
  primary_shared_token: 'Shared term',
  shared_tokens: 'Shared terms',
  cluster_id: 'Cluster',
}

const EVIDENCE_FIELD_ORDER = Object.keys(EVIDENCE_FIELD_LABELS)

/**
 * Structured evidence for one edge, rendered from the payload already in hand \u2014
 * no network request, so it appears instantly on click.
 */
function EvidencePanel({ item }) {
  const evidence = item?.evidence || {}
  const passLabel = PASS_LABELS[evidence.pass]
  const rows = EVIDENCE_FIELD_ORDER
    .filter(key => evidence[key] !== undefined && evidence[key] !== null && evidence[key] !== '')
    .map(key => {
      const raw = evidence[key]
      const value = Array.isArray(raw)
        ? raw.join(', ')
        : typeof raw === 'number'
          ? String(Math.round(raw * 1000) / 1000)
          : String(raw)
      return { key, label: EVIDENCE_FIELD_LABELS[key], value }
    })

  if (!passLabel && rows.length === 0) return null

  return (
    <div className="mt-2 rounded-lg border border-white/[0.06] bg-black/25 px-2.5 py-2 space-y-1.5">
      {passLabel && (
        <div className="text-[10px] uppercase tracking-[0.12em] text-gray-500">
          {passLabel}
        </div>
      )}
      {evidence.synonym_match && (
        <div className="text-[10px] text-amber-300/90">Matched as a synonym</div>
      )}
      {rows.map(row => (
        <div key={row.key} className="flex items-start justify-between gap-2 text-[11px]">
          <span className="text-gray-500 shrink-0">{row.label}</span>
          <span className="text-gray-300 text-right break-words">{row.value}</span>
        </div>
      ))}
    </div>
  )
}

/**
 * "Explain in plain language" \u2014 the only place an LLM call can happen in this
 * view, and only when the user asks for it. The answer is cached per edge
 * server-side, and works offline via the deterministic mock provider.
 */
function ExplainEdgeButton({ sessionId, item }) {
  const [state, setState] = useState({ status: 'idle', text: '', error: '' })

  const canExplain =
    sessionId != null &&
    Number.isFinite(Number(item?.edgeSource)) &&
    Number.isFinite(Number(item?.edgeTarget))

  if (!canExplain) return null

  const handleClick = async () => {
    setState({ status: 'loading', text: '', error: '' })
    try {
      const data = await explainEdge(
        sessionId, { source: Number(item.edgeSource), target: Number(item.edgeTarget) },
      )
      setState({ status: 'done', text: data.explanation || '', error: '' })
    } catch (err) {
      setState({ status: 'error', text: '', error: getErrorMessage(err) })
    }
  }

  return (
    <div className="mt-2">
      {state.status !== 'done' && (
        <button
          type="button"
          onClick={handleClick}
          disabled={state.status === 'loading'}
          className="inline-flex items-center gap-1 text-[11px] font-semibold text-brand-300 hover:text-brand-200 disabled:opacity-50"
        >
          <Sparkles size={11} />
          {state.status === 'loading' ? 'Explaining...' : 'Explain in plain language'}
        </button>
      )}
      {state.status === 'done' && (
        <p className="text-[11px] text-gray-300 leading-relaxed whitespace-normal break-words">
          {state.text}
        </p>
      )}
      {state.status === 'error' && (
        <p className="mt-1 text-[11px] text-red-400">{state.error}</p>
      )}
    </div>
  )
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
      const art = extractArtifactName(item)
      if (art) produced.push({ art, target: item.node_id })
    }
  })
  ;(parents || []).forEach(item => {
    if (item.relation === 'data') {
      const art = extractArtifactName(item)
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

function RelatedList({ title, items, sessionId }) {
  if (!items?.length) return null
  return (
    <div>
      <div className="text-[11px] uppercase tracking-[0.12em] text-gray-500 mb-1">{title}</div>
      <div className="space-y-1.5">
        {items.slice(0, 5).map(item => {
          const color = RELATION_COLORS[item.relation] || '#94a3b8'
          const art = item.relation === 'data' ? extractArtifactName(item) : null
          return (
            <div key={`${title}-${item.node_id}-${item.relation}`} className="rounded-lg bg-black/20 border border-white/[0.05] px-2.5 py-2">
              <div className="flex items-center justify-between gap-2">
                <span className="font-mono text-xs text-brand-300">{item.node_id}</span>
                <span className="flex items-center gap-1.5">
                  <ConfidenceBadge tier={item.confidenceTier} />
                  <span
                    className="text-[10px] px-1.5 py-0.5 rounded-full font-medium"
                    style={{ background: color + '22', color }}
                  >
                    {item.relation}
                  </span>
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
              <EvidencePanel item={item} />
              <ExplainEdgeButton sessionId={sessionId} item={item} />
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default function DependencyNodeDetailsPanel({ node, sessionId = null }) {
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

      <RelatedList title="Parent preview" items={node.parents} sessionId={sessionId} />
      <RelatedList title="Child preview" items={node.children} sessionId={sessionId} />
    </div>
  )
}
