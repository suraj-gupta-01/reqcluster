import { useMemo, useState } from 'react'
import { AlertTriangle, ChevronDown, ChevronLeft, ChevronRight, Search } from 'lucide-react'

const PAGE_SIZE = 25

function Badge({ children, className = '' }) {
  return <span className={`badge ${className}`}>{children}</span>
}

function confidenceClass(confidence) {
  if (confidence === null || confidence === undefined) return 'bg-gray-800 text-gray-400'
  if (confidence >= 0.8) return 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/20'
  if (confidence >= 0.7) return 'bg-amber-500/15 text-amber-300 border border-amber-500/20'
  return 'bg-red-500/15 text-red-300 border border-red-500/20'
}

function metricValue(report, key) {
  const value = report?.[key]
  if (value === null || value === undefined) return '-'
  return typeof value === 'number' ? value.toFixed(3) : String(value)
}

export default function EnrichmentResultsTable({ results, requirements = [] }) {
  const [search, setSearch] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [lowConfidenceOnly, setLowConfidenceOnly] = useState(false)
  const [warningsOnly, setWarningsOnly] = useState(false)
  const [page, setPage] = useState(1)
  const [expanded, setExpanded] = useState({})

  const requirementMap = useMemo(() => {
    const map = {}
    requirements.forEach(req => { map[req.req_id] = req })
    return map
  }, [requirements])

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    return (results || []).filter((row) => {
      const req = requirementMap[row.requirement_id]
      const text = `${row.requirement_id || ''} ${row.expanded_text || ''} ${req?.text || ''}`.toLowerCase()
      if (q && !text.includes(q)) return false
      if (statusFilter !== 'all' && row.status !== statusFilter) return false
      if (lowConfidenceOnly && !((row.confidence ?? 1) < 0.7)) return false
      if (warningsOnly && !(row.warnings?.length > 0)) return false
      return true
    })
  }, [results, requirementMap, search, statusFilter, lowConfidenceOnly, warningsOnly])

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
  const safePage = Math.min(page, totalPages)
  const visible = filtered.slice((safePage - 1) * PAGE_SIZE, safePage * PAGE_SIZE)

  if (!results || results.length === 0) {
    return (
      <div className="card p-6 text-center text-sm text-gray-500">
        No enrichment results yet. Start enrichment for this session.
      </div>
    )
  }

  return (
    <div className="card overflow-hidden">
      <div className="p-4 border-b border-gray-800 flex flex-wrap gap-3 items-center">
        <div className="relative flex-1 min-w-64">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
          <label className="sr-only" htmlFor="enrichment-search">Search enrichment results</label>
          <input
            id="enrichment-search"
            type="text"
            placeholder="Search requirement ID or text"
            value={search}
            onChange={e => { setSearch(e.target.value); setPage(1) }}
            className="input text-sm pl-9 w-full"
          />
        </div>
        <label className="sr-only" htmlFor="enrichment-status-filter">Filter by status</label>
        <select
          id="enrichment-status-filter"
          value={statusFilter}
          onChange={e => { setStatusFilter(e.target.value); setPage(1) }}
          className="input text-sm"
        >
          <option value="all">All statuses</option>
          <option value="succeeded">Succeeded</option>
          <option value="failed">Failed</option>
          <option value="pending">Pending</option>
        </select>
        <label className="flex items-center gap-2 text-sm text-gray-400">
          <input type="checkbox" checked={lowConfidenceOnly} onChange={e => { setLowConfidenceOnly(e.target.checked); setPage(1) }} />
          Low confidence
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-400">
          <input type="checkbox" checked={warningsOnly} onChange={e => { setWarningsOnly(e.target.checked); setPage(1) }} />
          Warnings
        </label>
      </div>

      <div className="divide-y divide-gray-800/60">
        {visible.length === 0 ? (
          <div className="py-10 text-center text-sm text-gray-500">No enrichment rows match your filters.</div>
        ) : visible.map((row, index) => {
          const key = row.requirement_id || `${safePage}-${index}`
          const isOpen = !!expanded[key]
          const req = requirementMap[row.requirement_id]
          const warningCount = row.warnings?.length || 0
          return (
            <div key={key} className="hover:bg-gray-800/30">
              <button
                type="button"
                onClick={() => setExpanded(prev => ({ ...prev, [key]: !prev[key] }))}
                className="w-full text-left px-4 py-3 flex items-center gap-4"
              >
                <ChevronDown size={14} className={`text-gray-500 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
                <span className="font-mono text-xs text-brand-400 w-24 flex-shrink-0">{row.requirement_id || '-'}</span>
                <span className="text-sm text-gray-300 line-clamp-1 flex-1">
                  {row.expanded_text || 'No expanded text persisted.'}
                </span>
                <Badge className={confidenceClass(row.confidence)}>
                  {row.confidence === null || row.confidence === undefined ? '-' : row.confidence.toFixed(2)}
                </Badge>
                <Badge className={warningCount ? 'bg-amber-500/15 text-amber-300 border border-amber-500/20' : 'bg-gray-800 text-gray-500'}>
                  {warningCount} warnings
                </Badge>
                <Badge className={row.status === 'succeeded' ? 'bg-emerald-500/15 text-emerald-300' : 'bg-red-500/15 text-red-300'}>
                  {row.status}
                </Badge>
              </button>

              {isOpen && (
                <div className="px-12 pb-5 grid lg:grid-cols-2 gap-4">
                  <div className="space-y-4">
                    <div>
                      <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-1">Original Requirement</div>
                      <p className="text-sm text-gray-300 leading-relaxed">{req?.text || 'Original text is unavailable in this view.'}</p>
                    </div>
                    <div>
                      <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-1">Expanded Requirement</div>
                      <p className="text-sm text-gray-200 leading-relaxed">{row.expanded_text || '-'}</p>
                    </div>
                    <div>
                      <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-1">Functional Intent</div>
                      <p className="text-sm text-gray-300">{row.functional_intent || '-'}</p>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <TagList title="Domain Terms" items={row.domain_terms} color="brand" />
                    <TagList title="Mentioned Components" items={row.mentioned_components} color="emerald" />
                    <TagList title="Assumptions" items={row.assumptions} color="gray" />

                    <div>
                      <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-1">Quality Metrics</div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <Metric label="Lexical overlap" value={metricValue(row.quality_report, 'lexical_overlap')} />
                        <Metric label="Length ratio" value={metricValue(row.quality_report, 'length_ratio')} />
                        <Metric label="Risk score" value={metricValue(row.quality_report, 'hallucination_risk_score')} />
                        <Metric label="Adjusted confidence" value={metricValue(row.quality_report, 'adjusted_confidence_score')} />
                      </div>
                    </div>

                    {warningCount > 0 && (
                      <div className="space-y-1">
                        {row.warnings.map((warning) => (
                          <div key={warning} className="flex items-start gap-2 text-xs text-amber-300">
                            <AlertTriangle size={12} className="mt-0.5 flex-shrink-0" />
                            <span>{warning}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>

      <div className="px-4 py-3 border-t border-gray-800 flex items-center justify-between text-xs text-gray-500">
        <span>Showing {visible.length} of {filtered.length} filtered rows ({results.length} total)</span>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setPage(value => Math.max(1, value - 1))}
            disabled={safePage === 1}
            className="btn-secondary text-xs py-1 px-2 disabled:opacity-40"
          >
            <ChevronLeft size={13} />
          </button>
          <span>Page {safePage} of {totalPages}</span>
          <button
            type="button"
            onClick={() => setPage(value => Math.min(totalPages, value + 1))}
            disabled={safePage === totalPages}
            className="btn-secondary text-xs py-1 px-2 disabled:opacity-40"
          >
            <ChevronRight size={13} />
          </button>
        </div>
      </div>
    </div>
  )
}

function TagList({ title, items = [], color }) {
  const styles = {
    brand: 'bg-brand-600/15 text-brand-300 border-brand-500/20',
    emerald: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/20',
    gray: 'bg-gray-800 text-gray-400 border-gray-700',
  }
  return (
    <div>
      <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-1">{title}</div>
      {items?.length ? (
        <div className="flex flex-wrap gap-1.5">
          {items.map(item => <span key={item} className={`badge border ${styles[color]}`}>{item}</span>)}
        </div>
      ) : (
        <span className="text-xs text-gray-600">None</span>
      )}
    </div>
  )
}

function Metric({ label, value }) {
  return (
    <div className="bg-gray-950/50 border border-gray-800 rounded-lg p-2">
      <div className="text-gray-300">{value}</div>
      <div className="text-gray-600">{label}</div>
    </div>
  )
}
