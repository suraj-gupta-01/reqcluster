import { useMemo, useState } from 'react'
import { Tag } from 'lucide-react'

const DEFAULT_LIMIT = 24

function normalizeTerms(terms) {
  if (!Array.isArray(terms)) return []
  return terms
    .map((item) => {
      if (typeof item === 'string') return { term: item, score: null }
      if (item && typeof item === 'object') return { term: item.term || item.label || '', score: item.score ?? null }
      return { term: '', score: null }
    })
    .filter(item => item.term)
}

export default function DomainVocabularyTags({ terms, limit = DEFAULT_LIMIT }) {
  const [expanded, setExpanded] = useState(false)
  const normalized = useMemo(() => normalizeTerms(terms), [terms])
  const visible = expanded ? normalized : normalized.slice(0, limit)

  if (normalized.length === 0) {
    return (
      <div className="card p-4 text-sm text-gray-500">
        No domain vocabulary available yet.
      </div>
    )
  }

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Tag size={15} className="text-brand-400" />
          <h2 className="text-sm font-semibold text-gray-200">Domain Vocabulary</h2>
        </div>
        <span className="text-xs text-gray-500">{normalized.length} terms</span>
      </div>
      <div className="flex flex-wrap gap-2">
        {visible.map((item) => (
          <span
            key={item.term}
            title={item.score !== null ? `Score: ${item.score}` : item.term}
            className="badge bg-brand-600/15 text-brand-300 border border-brand-500/20"
          >
            {item.term}
          </span>
        ))}
      </div>
      {normalized.length > limit && (
        <button
          type="button"
          onClick={() => setExpanded(value => !value)}
          className="mt-3 text-xs text-brand-400 hover:text-brand-300"
        >
          {expanded ? 'Show less' : `Show ${normalized.length - limit} more`}
        </button>
      )}
    </div>
  )
}
