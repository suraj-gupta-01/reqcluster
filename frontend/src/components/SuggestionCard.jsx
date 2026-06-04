import { useState } from 'react'
import { GitMerge, Scissors, ChevronDown, ChevronUp, Check, X, Loader } from 'lucide-react'
import ClusterComparisonPanel from './ClusterComparisonPanel.jsx'
import RepresentativesList from './RepresentativesList.jsx'

const STATUS_COLORS = {
  pending: 'bg-amber-500/20 text-amber-400',
  accepted: 'bg-emerald-500/20 text-emerald-400',
  rejected: 'bg-red-500/20 text-red-400',
  applied: 'bg-brand-500/20 text-brand-400',
}

function ScoreBadge({ label, value, good }) {
  if (value == null) return null
  const formatted = typeof value === 'number' ? value.toFixed(3) : value
  return (
    <div className="text-center">
      <div className={`text-sm font-mono font-bold ${good ? 'text-emerald-400' : 'text-gray-300'}`}>
        {formatted}
      </div>
      <div className="text-xs text-gray-500">{label}</div>
    </div>
  )
}

export default function SuggestionCard({
  suggestion,
  clusterInfo = {},
  onAccept,
  onReject,
  isApplying = false,
}) {
  const [expanded, setExpanded] = useState(false)

  const isMerge = suggestion.suggestion_type === 'merge'
  const isPending = suggestion.status === 'pending'

  const Icon = isMerge ? GitMerge : Scissors
  const title = isMerge
    ? `Merge: ${suggestion.cluster_a_label || 'A'} + ${suggestion.cluster_b_label || 'B'}`
    : `Split: ${suggestion.cluster_label || `Cluster ${suggestion.cluster_id}`}`

  return (
    <div className="card overflow-hidden transition-all duration-200">
      {/* Collapsed header */}
      <div
        className="px-4 py-3 flex items-center gap-3 cursor-pointer hover:bg-gray-800/50 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <Icon size={16} className={isMerge ? 'text-blue-400' : 'text-orange-400'} />

        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium text-gray-200 truncate">{title}</div>
          <div className="text-xs text-gray-500 mt-0.5">
            {isMerge
              ? `Similarity: ${(suggestion.similarity_score ?? 0).toFixed(3)}`
              : `Bimodality: ${(suggestion.bimodality_score ?? 0).toFixed(3)}`}
            {suggestion.silhouette_delta != null && (
              <span className={suggestion.silhouette_delta > 0 ? ' text-emerald-400' : ' text-gray-500'}>
                {' · '}Δ silhouette: {suggestion.silhouette_delta > 0 ? '+' : ''}{suggestion.silhouette_delta.toFixed(4)}
              </span>
            )}
          </div>
        </div>

        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${STATUS_COLORS[suggestion.status] || STATUS_COLORS.pending}`}>
          {suggestion.status}
        </span>

        {expanded ? <ChevronUp size={14} className="text-gray-500" /> : <ChevronDown size={14} className="text-gray-500" />}
      </div>

      {/* Expanded details */}
      {expanded && (
        <div className="px-4 pb-4 border-t border-gray-800/50 pt-3 space-y-4">
          {/* Scores */}
          <div className="flex gap-6 flex-wrap">
            {isMerge && (
              <ScoreBadge label="Similarity" value={suggestion.similarity_score} good={suggestion.similarity_score >= 0.8} />
            )}
            {!isMerge && (
              <>
                <ScoreBadge label="Bimodality" value={suggestion.bimodality_score} good={suggestion.bimodality_score >= 0.5} />
                <ScoreBadge label="Spread" value={suggestion.spread_score} good={suggestion.spread_score >= 0.3} />
              </>
            )}
            <ScoreBadge
              label="Δ Silhouette"
              value={suggestion.silhouette_delta}
              good={suggestion.silhouette_delta > 0}
            />
            {suggestion.coherence_score != null && (
              <ScoreBadge label="Coherence" value={suggestion.coherence_score} good={suggestion.coherence_score >= 0.7} />
            )}
          </div>

          {/* Rationale */}
          {suggestion.rationale && (
            <div>
              <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-1">Rationale</div>
              <p className="text-sm text-gray-300 leading-relaxed">{suggestion.rationale}</p>
            </div>
          )}

          {/* Cluster comparison */}
          <ClusterComparisonPanel suggestion={suggestion} clusterInfo={clusterInfo} />

          {/* Representatives */}
          <RepresentativesList reqIds={suggestion.representative_req_ids} />

          {/* Summary */}
          {suggestion.summary && (
            <div>
              <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-1">Cluster Summary</div>
              <p className="text-sm text-gray-400 italic">{suggestion.summary}</p>
            </div>
          )}

          {/* Actions */}
          {isPending && (
            <div className="flex gap-2 pt-2 border-t border-gray-800/50">
              <button
                onClick={(e) => { e.stopPropagation(); onAccept?.(suggestion.id) }}
                disabled={isApplying}
                className="btn-primary flex items-center gap-1.5 text-sm px-4 py-1.5 disabled:opacity-50"
              >
                {isApplying ? <Loader size={14} className="animate-spin" /> : <Check size={14} />}
                Accept
              </button>
              <button
                onClick={(e) => { e.stopPropagation(); onReject?.(suggestion.id) }}
                disabled={isApplying}
                className="btn-secondary flex items-center gap-1.5 text-sm px-4 py-1.5 disabled:opacity-50"
              >
                <X size={14} />
                Reject
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
