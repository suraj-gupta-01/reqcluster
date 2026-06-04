import { AlertTriangle, BarChart3 } from 'lucide-react'

function fmt(value) {
  if (value === null || value === undefined) return '-'
  if (Array.isArray(value)) return value.join(' x ')
  return typeof value === 'number' ? Number(value).toFixed(4) : value
}

function VariantCard({ title, data }) {
  return (
    <div className="bg-gray-950/50 border border-gray-800 rounded-lg p-3">
      <div className="text-sm font-semibold text-gray-200 mb-3">{title}</div>
      <div className="grid grid-cols-2 gap-2 text-xs">
        <span className="text-gray-500">Embedding shape</span><span className="text-gray-300 text-right">{fmt(data?.embedding_shape)}</span>
        <span className="text-gray-500">Clusters</span><span className="text-gray-300 text-right">{fmt(data?.n_clusters)}</span>
        <span className="text-gray-500">Noise count</span><span className="text-gray-300 text-right">{fmt(data?.noise_count)}</span>
        <span className="text-gray-500">Noise rate</span><span className="text-gray-300 text-right">{fmt(data?.noise_rate)}</span>
        <span className="text-gray-500">Silhouette</span><span className="text-gray-300 text-right">{fmt(data?.silhouette_score_10d)}</span>
      </div>
    </div>
  )
}

export default function AblationReportPanel({ report }) {
  if (!report) {
    return (
      <div className="card p-4 text-sm text-gray-500">
        No ablation. Ablation report will appear after clustering with ablation enabled.
      </div>
    )
  }

  const base = report.base || {}
  const enriched = report.enriched || {}
  const silhouetteDelta = enriched.silhouette_score_10d !== null && base.silhouette_score_10d !== null
    ? enriched.silhouette_score_10d - base.silhouette_score_10d
    : null
  const noiseDelta = enriched.noise_rate !== null && base.noise_rate !== null
    ? enriched.noise_rate - base.noise_rate
    : null
  const clusterDelta = (enriched.n_clusters ?? 0) - (base.n_clusters ?? 0)

  return (
    <div className="card p-4">
      <div className="flex items-center gap-2 mb-4">
        <BarChart3 size={15} className="text-brand-400" />
        <h2 className="text-sm font-semibold text-gray-200">Ablation Report</h2>
        <span className="badge bg-gray-800 text-gray-400">{report.mode || 'hybrid'}</span>
      </div>

      <div className="grid md:grid-cols-2 gap-3">
        <VariantCard title="Base" data={base} />
        <VariantCard title="Enriched / Hybrid" data={enriched} />
      </div>

      <div className="grid grid-cols-3 gap-3 mt-3">
        {[
          ['Cluster delta', clusterDelta],
          ['Noise rate delta', noiseDelta],
          ['Silhouette delta', silhouetteDelta],
        ].map(([label, value]) => (
          <div key={label} className="bg-gray-950/50 border border-gray-800 rounded-lg p-3">
            <div className="text-lg font-semibold text-white">{fmt(value)}</div>
            <div className="text-xs text-gray-500">{label}</div>
          </div>
        ))}
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
