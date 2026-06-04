import { AlertTriangle, GitCompare } from 'lucide-react'

function fmt(value) {
  return value === null || value === undefined ? '-' : Number(value).toFixed(4)
}

export default function EmbeddingComparisonPanel({ report }) {
  if (!report) {
    return (
      <div className="card p-4 text-sm text-gray-500">
        No comparison. Embedding comparison will appear after clustering with comparison enabled.
      </div>
    )
  }

  const aggregate = report.aggregate || {}
  const thresholds = report.delta_threshold_counts || {}
  const nearest = report.nearest_neighbor_preservation || {}

  return (
    <div className="card p-4">
      <div className="flex items-start justify-between mb-4 gap-3">
        <div>
          <div className="flex items-center gap-2">
            <GitCompare size={15} className="text-brand-400" />
            <h2 className="text-sm font-semibold text-gray-200">Embedding Comparison</h2>
          </div>
          <p className="text-xs text-gray-500 mt-1">
            These metrics compare base embeddings with enriched/hybrid embeddings.
          </p>
        </div>
        <span className="text-xs text-gray-500">{report.n_requirements ?? 0} requirements</span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          ['Mean cosine', aggregate.mean_cosine_similarity],
          ['Median cosine', aggregate.median_cosine_similarity],
          ['Min cosine', aggregate.min_cosine_similarity],
          ['Max cosine', aggregate.max_cosine_similarity],
          ['Mean delta', aggregate.mean_delta],
          ['NN preservation', nearest.score],
          ['Delta > 0.05', thresholds['0.05'] ?? thresholds['0.050000']],
          ['Delta > 0.10', thresholds['0.10'] ?? thresholds['0.100000']],
        ].map(([label, value]) => (
          <div key={label} className="bg-gray-950/50 border border-gray-800 rounded-lg p-3">
            <div className="text-lg font-semibold text-white">{typeof value === 'number' ? fmt(value) : value ?? '-'}</div>
            <div className="text-xs text-gray-500">{label}</div>
          </div>
        ))}
      </div>

      <div className="mt-3 text-xs text-gray-500">
        Delta &gt; 0.20: {thresholds['0.20'] ?? thresholds['0.200000'] ?? 0}
      </div>

      {report.warnings?.length > 0 && (
        <div className="mt-4 space-y-1">
          {report.warnings.map((warning) => (
            <div key={warning} className="flex items-start gap-2 text-xs text-amber-300">
              <AlertTriangle size={12} className="mt-0.5 flex-shrink-0" />
              <span>{warning}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
